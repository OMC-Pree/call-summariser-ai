-- Athena DDL for Call Summarizer - Partitioned Tables
-- Run these queries in Athena Console or via AWS CLI

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS call_summaries;

-- Main summaries table with partition projection
CREATE EXTERNAL TABLE IF NOT EXISTS call_summaries.summaries (
    summary_schema_version string,
    model_version string,
    prompt_version string,
    meeting struct<
        id: string,
        employerName: string,
        coach: string,
        createdAt: string
    >,
    sentiment struct<
        label: string,
        confidence: double,
        evidence_spans: array<string>
    >,
    themes array<struct<
        id: string,
        label: string,
        group: string,
        confidence: double,
        evidence_quote: string
    >>,
    summary string,
    actions array<struct<
        id: string,
        text: string,
        due: string
    >>,
    call_metadata struct<
        source: string,
        saved_at: string,
        insights_version: string,
        prompt_version: string,
        model_version: string,
        schema_version: string,
        prefix_version: string
    >,
    insights struct<
        action_count: int,
        theme_count: int,
        sentiment_label: string,
        is_escalation_candidate: boolean,
        risk_flags: array<string>,
        categories: array<string>
    >
)
PARTITIONED BY (
    version string,
    year int,
    month int
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
STORED AS INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat'
OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION 's3://call-summariser-summarybucket-3wtnjhb9vvq0/summaries/'
TBLPROPERTIES (
    'has_encrypted_data'='false',
    'projection.enabled'='true',
    'projection.version.type'='enum',
    'projection.version.values'='1.0,1.1',
    'projection.year.type'='integer',
    'projection.year.range'='2024,2030',
    'projection.year.interval'='1',
    'projection.month.type'='integer',
    'projection.month.range'='1,12',
    'projection.month.interval'='1',
    'projection.month.digits'='2',
    'storage.location.template'='s3://call-summariser-summarybucket-3wtnjhb9vvq0/summaries/version=${version}/year=${year}/month=${month}/'
);

-- View for latest summaries only (most commonly used)
CREATE OR REPLACE VIEW call_summaries.latest_summaries AS
SELECT
    meeting.id as meeting_id,
    summary,
    actions,
    themes,
    meeting.coach as coach_name,
    sentiment.label as overall_sentiment,
    sentiment.confidence as sentiment_confidence,
    model_version,
    prompt_version,
    meeting.createdAt as created_at,
    insights.action_count,
    insights.theme_count,
    cardinality(actions) as action_count_calc,
    cardinality(themes) as theme_count_calc,
    version,
    year,
    month
FROM call_summaries.summaries
WHERE version = '1.1'  -- Current version
    AND year >= year(current_date) - 1  -- Last year and current
ORDER BY meeting.createdAt DESC;

-- View for version comparison analysis
CREATE OR REPLACE VIEW call_summaries.version_comparison AS
SELECT
    meeting.id as meeting_id,
    version,
    model_version,
    prompt_version,
    sentiment.confidence as sentiment_confidence,
    cardinality(actions) as action_count,
    cardinality(themes) as theme_count,
    meeting.createdAt as created_at,
    year,
    month
FROM call_summaries.summaries
WHERE meeting.id IN (
    -- Only meetings that have multiple versions
    SELECT meeting.id
    FROM call_summaries.summaries
    GROUP BY meeting.id
    HAVING count(DISTINCT version) > 1
)
ORDER BY meeting.id, version, meeting.createdAt;

-- Performance analytics view
CREATE OR REPLACE VIEW call_summaries.performance_metrics AS
SELECT
    version,
    model_version,
    prompt_version,
    year,
    month,
    count(*) as total_summaries,
    avg(cardinality(actions)) as avg_actions,
    avg(cardinality(themes)) as avg_themes,
    avg(sentiment.confidence) as avg_sentiment_confidence,
    count(DISTINCT meeting.id) as unique_meetings,
    count(DISTINCT meeting.coach) as unique_coaches
FROM call_summaries.summaries
GROUP BY version, model_version, prompt_version, year, month
ORDER BY year DESC, month DESC, version;

-- Coach performance view
CREATE OR REPLACE VIEW call_summaries.coach_analytics AS
SELECT
    meeting.coach as coach_name,
    version,
    year,
    month,
    count(*) as meeting_count,
    avg(cardinality(actions)) as avg_actions_per_meeting,
    avg(cardinality(themes)) as avg_themes_per_meeting,
    avg(sentiment.confidence) as avg_sentiment_confidence,
    array_join(
        array_agg(DISTINCT sentiment.label),
        ', '
    ) as sentiment_distribution
FROM call_summaries.summaries
WHERE meeting.coach IS NOT NULL
GROUP BY meeting.coach, version, year, month
ORDER BY meeting.coach, year DESC, month DESC;

-- Monthly trends view for dashboard
CREATE OR REPLACE VIEW call_summaries.monthly_trends AS
SELECT
    year,
    month,
    version,
    count(*) as summary_count,
    count(DISTINCT meeting.id) as unique_meetings,
    count(DISTINCT meeting.coach) as active_coaches,
    avg(cardinality(actions)) as avg_actions,
    avg(cardinality(themes)) as avg_themes,
    avg(sentiment.confidence) as avg_sentiment_confidence,
    sum(CASE WHEN sentiment.label = 'Positive' THEN 1 ELSE 0 END) as positive_meetings,
    sum(CASE WHEN sentiment.label = 'Negative' THEN 1 ELSE 0 END) as negative_meetings,
    sum(CASE WHEN sentiment.label = 'Neutral' THEN 1 ELSE 0 END) as neutral_meetings
FROM call_summaries.summaries
GROUP BY year, month, version
ORDER BY year DESC, month DESC, version;