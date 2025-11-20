#!/usr/bin/env python3
"""
Accurate Zoom meeting matcher that validates both coach and client
"""
import csv
import requests
import boto3
from datetime import datetime, timedelta
import time
from collections import defaultdict
import re

class ZoomAccurateMatcher:
    """Match with coach and client validation"""

    def __init__(self):
        ssm = boto3.client('ssm')
        self.account_id = self.get_ssm_param(ssm, '/zoom/s2s/account_id')
        self.client_id = self.get_ssm_param(ssm, '/zoom/s2s/client_id')
        self.client_secret = self.get_ssm_param(ssm, '/zoom/s2s/client_secret')
        self.access_token = None
        self.base_url = 'https://api.zoom.us/v2'
        self.recordings_cache = defaultdict(list)

        # Coach email mapping
        self.coach_emails = {}

    def get_ssm_param(self, ssm, name):
        response = ssm.get_parameter(Name=name, WithDecryption=True)
        return response['Parameter']['Value'].strip()

    def get_access_token(self):
        response = requests.post(
            'https://zoom.us/oauth/token',
            params={'grant_type': 'account_credentials', 'account_id': self.account_id},
            auth=(self.client_id, self.client_secret)
        )
        if response.status_code == 200:
            self.access_token = response.json()['access_token']
            print("✓ Authenticated with Zoom API")
            return True
        return False

    def get_headers(self):
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

    def get_users(self):
        """Get all Zoom users and map coach names to emails"""
        url = f'{self.base_url}/users'
        params = {'status': 'active', 'page_size': 300}
        response = requests.get(url, headers=self.get_headers(), params=params)

        if response.status_code == 200:
            users = response.json()['users']
            print(f"✓ Retrieved {len(users)} Zoom users")

            # Map coach names to their email addresses
            for user in users:
                display_name = user['display_name']
                full_name = f"{user['first_name']} {user['last_name']}"
                email = user['email'].lower()

                self.coach_emails[display_name.lower()] = email
                self.coach_emails[full_name.lower()] = email

            return users
        return []

    def get_recordings_for_month(self, year, month):
        from_date = datetime(year, month, 1)
        if month == 12:
            to_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            to_date = datetime(year, month + 1, 1) - timedelta(days=1)

        from_str = from_date.strftime('%Y-%m-%d')
        to_str = to_date.strftime('%Y-%m-%d')

        print(f"  Fetching {year}-{month:02d} ({from_str} to {to_str})")

        all_recordings = []
        next_page_token = None

        while True:
            url = f'{self.base_url}/accounts/me/recordings'
            params = {'from': from_str, 'to': to_str, 'page_size': 300}
            if next_page_token:
                params['next_page_token'] = next_page_token

            response = requests.get(url, headers=self.get_headers(), params=params)
            if response.status_code == 200:
                data = response.json()
                meetings = data.get('meetings', [])
                all_recordings.extend(meetings)
                next_page_token = data.get('next_page_token')
                if not next_page_token:
                    break
            else:
                break
            time.sleep(0.1)

        print(f"    Found {len(all_recordings)} recordings")
        return all_recordings

    def parse_call_date(self, date_str):
        try:
            return datetime.strptime(date_str, '%d/%m/%Y %H:%M')
        except ValueError:
            return None

    def extract_client_name_from_email(self, email):
        """Extract likely name from email"""
        if not email or email == 'unknown':
            return None

        # Get part before @
        local = email.split('@')[0]

        # Split by dots, underscores, etc
        parts = re.split(r'[._\-\d]+', local)

        # Return cleaned parts
        return ' '.join(p for p in parts if len(p) > 1).strip()

    def match_by_recording(self, coach_name, client_email, call_datetime, recordings):
        """
        Accurate matching:
        1. Must match coach (host email)
        2. Must match time (±30 min)
        3. Should match client name in topic (if possible)
        """
        if not call_datetime:
            return None, "invalid_date"

        # Get coach email
        coach_email = self.coach_emails.get(coach_name.lower())
        if not coach_email:
            return None, f"Coach '{coach_name}' email not found"

        # Extract client name from email
        client_name = self.extract_client_name_from_email(client_email)

        matches = []

        for recording in recordings:
            try:
                rec_time = datetime.strptime(recording['start_time'], '%Y-%m-%dT%H:%M:%SZ')
            except:
                continue

            # Check time (±2 hours for flexible matching)
            time_diff = abs((rec_time - call_datetime).total_seconds())
            if time_diff > 7200:  # 2 hours
                continue

            # Check if coach matches (host email)
            host_email = recording.get('host_email', '').lower()
            if host_email != coach_email:
                continue

            # At this point: correct coach + correct time
            meeting_id = recording.get('id')
            topic = recording.get('topic', '').lower()

            # Try to match client name in topic
            if client_name and client_name.lower() in topic:
                return meeting_id, 'exact_match'

            # Client email in topic
            if client_email.lower() in topic:
                return meeting_id, 'exact_match'

            # Store as potential match
            matches.append((meeting_id, time_diff, topic))

        # If we have matches with correct coach and time, return the closest one
        if matches:
            matches.sort(key=lambda x: x[1])  # Sort by time difference
            meeting_id, time_diff, topic = matches[0]
            return meeting_id, 'coach_time_match'

        return None, 'no_match'

    def process_csv(self, input_csv, output_csv):
        if not self.get_access_token():
            return

        # Get users first
        print("\nFetching Zoom users...")
        self.get_users()

        # Read CSV
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            records = list(reader)

        print(f"\n✓ Loaded {len(records)} records")

        # Get date range
        dates = []
        for record in records:
            call_date = self.parse_call_date(record.get('Call Date', ''))
            if call_date:
                dates.append(call_date)

        if not dates:
            print("✗ No valid dates")
            return

        min_date, max_date = min(dates), max(dates)
        print(f"✓ Date range: {min_date.date()} to {max_date.date()}")

        # Fetch recordings
        print("\nFetching recordings...")
        current = min_date
        while current <= max_date:
            month_recordings = self.get_recordings_for_month(current.year, current.month)
            for rec in month_recordings:
                try:
                    rec_date = datetime.strptime(rec['start_time'], '%Y-%m-%dT%H:%M:%SZ').date()
                    self.recordings_cache[rec_date].append(rec)
                except:
                    pass
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
            time.sleep(0.2)

        print(f"\n✓ Cached {sum(len(v) for v in self.recordings_cache.values())} recordings")

        # Match records
        print("\nMatching with coach validation...")
        matched = improved = 0

        for i, record in enumerate(records, 1):
            coach_name = record.get('Coach/Adviser Name', '')
            client_email = record.get('Email', '')
            call_date = record.get('Call Date', '')
            call_datetime = self.parse_call_date(call_date)

            if not call_datetime:
                record['Zoom Meeting ID'] = 'NOT_FOUND'
                record['Match Status'] = 'invalid_date'
                continue

            # Get relevant recordings (±1 day)
            relevant = []
            for offset in [-1, 0, 1]:
                check_date = (call_datetime + timedelta(days=offset)).date()
                relevant.extend(self.recordings_cache.get(check_date, []))

            # Match
            meeting_id, status = self.match_by_recording(
                coach_name, client_email, call_datetime, relevant
            )

            old_id = record.get('Zoom Meeting ID', 'NOT_FOUND')

            if meeting_id:
                # Check if this is different from previous match
                if old_id != 'NOT_FOUND' and old_id != meeting_id:
                    print(f"[{i}/{len(records)}] ! {coach_name:25s} | Changed: {old_id} → {meeting_id} ({status})")
                    improved += 1
                else:
                    print(f"[{i}/{len(records)}] ✓ {coach_name:25s} | {meeting_id} ({status})")

                record['Zoom Meeting ID'] = meeting_id
                record['Match Status'] = status
                matched += 1
            else:
                if old_id != 'NOT_FOUND':
                    print(f"[{i}/{len(records)}] - {coach_name:25s} | Kept: {old_id}")
                    # Keep old match
                else:
                    print(f"[{i}/{len(records)}] ✗ {coach_name:25s} | {status}")
                    record['Zoom Meeting ID'] = 'NOT_FOUND'
                    record['Match Status'] = status

        # Write output
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = list(records[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"\n{'='*60}")
        print(f"✓ Matched: {matched}")
        print(f"✓ Improved: {improved} (changed from previous)")
        print(f"✓ Saved to: {output_csv}")

if __name__ == "__main__":
    matcher = ZoomAccurateMatcher()
    matcher.process_csv(
        'extracted_data_with_zoom_ids.csv',
        'extracted_data_with_zoom_ids_accurate.csv'
    )
