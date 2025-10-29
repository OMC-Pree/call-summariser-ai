# CloudWatch Logs Insights - Cost Monitoring Queries

## Overview
These queries help you monitor and analyze AWS Bedrock costs for your call summariser application.

All logs include cost tracking in USD for both summary and case check operations.

---

## 1. Daily Cost Summary

Get total costs by day and operation type:

```
fields @timestamp, operation, cost_usd, input_tokens, output_tokens
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| stats
    sum(cost_usd) as total_cost,
    sum(input_tokens) as total_input_tokens,
    sum(output_tokens) as total_output_tokens,
    count(*) as num_calls
  by bin(@timestamp, 1d), operation
| sort @timestamp desc
```

**Use case:** Daily cost tracking and trend analysis

---

## 2. Cost Per Call Analysis

Calculate average, min, max cost per call:

```
fields @timestamp, meetingId, operation, cost_usd, input_tokens, output_tokens
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| stats
    avg(cost_usd) as avg_cost,
    min(cost_usd) as min_cost,
    max(cost_usd) as max_cost,
    avg(input_tokens) as avg_input_tokens,
    avg(output_tokens) as avg_output_tokens,
    count(*) as num_calls
  by operation
```

**Use case:** Understand typical costs and identify outliers

---

## 3. Total Costs by Meeting

Show complete cost per meeting (summary + case check):

```
fields @timestamp, meetingId, operation, cost_usd
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| stats
    sum(cost_usd) as total_cost,
    max(cost_usd) as max_operation_cost
  by meetingId
| sort total_cost desc
| limit 50
```

**Use case:** Identify most expensive calls

---

## 4. Hourly Cost Breakdown

Monitor costs by hour for capacity planning:

```
fields @timestamp, cost_usd, operation
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| stats
    sum(cost_usd) as hourly_cost,
    count(*) as num_operations
  by bin(@timestamp, 1h), operation
| sort @timestamp desc
| limit 100
```

**Use case:** Identify peak usage times and budget hourly costs

---

## 5. Cache Performance Analysis

Track cache hit rates and savings:

```
fields @timestamp, meetingId, operation,
       cache_read_tokens, cache_savings_usd, cost_usd
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| filter cache_read_tokens > 0
| stats
    sum(cache_savings_usd) as total_savings,
    avg(cache_savings_usd) as avg_savings_per_call,
    sum(cache_read_tokens) as total_cache_reads,
    count(*) as calls_with_cache
  by operation
```

**Use case:** Measure prompt caching effectiveness (when enabled)

---

## 6. Cost by Token Ranges

Segment calls by input size to understand cost drivers:

```
fields @timestamp, meetingId, input_tokens, output_tokens, cost_usd, operation
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| fields
    case
      when input_tokens < 10000 then "Small (< 10K)"
      when input_tokens < 20000 then "Medium (10K-20K)"
      when input_tokens < 30000 then "Large (20K-30K)"
      else "XLarge (> 30K)"
    end as size_category,
    cost_usd,
    operation
| stats
    avg(cost_usd) as avg_cost,
    count(*) as num_calls
  by size_category, operation
| sort size_category, operation
```

**Use case:** Understand how transcript length affects costs

---

## 7. Monthly Cost Projection

Calculate month-to-date and project monthly total:

```
fields @timestamp, cost_usd
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| filter @timestamp >= datefloor(@timestamp, 1M)
| stats
    sum(cost_usd) as mtd_cost,
    count(*) as mtd_calls,
    avg(cost_usd) as avg_cost_per_call
| fields
    mtd_cost,
    mtd_calls,
    avg_cost_per_call,
    mtd_cost / dateDiff(datefloor(@timestamp, 1M), now(), "day") * 30 as projected_monthly_cost
```

**Use case:** Budget forecasting and spend tracking

---

## 8. Cost Anomaly Detection

Find calls that cost significantly more than average:

```
fields @timestamp, meetingId, operation, cost_usd, input_tokens, output_tokens
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| stats avg(cost_usd) as avg_cost, stddev(cost_usd) as std_cost by operation
| fields operation, avg_cost, avg_cost + (2 * std_cost) as threshold
```

Then use threshold to find anomalies:

```
fields @timestamp, meetingId, operation, cost_usd, input_tokens, output_tokens, latency_ms
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| filter (operation = "summary" and cost_usd > 0.10) or (operation = "case_check" and cost_usd > 0.25)
| sort cost_usd desc
| limit 20
```

**Use case:** Identify unexpectedly expensive calls for investigation

---

## 9. Input vs Output Cost Breakdown

Compare input and output token costs:

```
fields @timestamp, operation,
       input_cost_usd, output_cost_usd, cost_usd,
       input_tokens, output_tokens
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| stats
    sum(input_cost_usd) as total_input_cost,
    sum(output_cost_usd) as total_output_cost,
    sum(cost_usd) as total_cost,
    sum(input_tokens) as total_input_tokens,
    sum(output_tokens) as total_output_tokens
  by operation
| fields
    operation,
    total_cost,
    total_input_cost,
    total_output_cost,
    total_input_cost / total_cost * 100 as input_cost_pct,
    total_output_cost / total_cost * 100 as output_cost_pct
```

**Use case:** Understand which tokens (input vs output) drive costs

---

## 10. Week-over-Week Cost Comparison

Compare costs across weeks:

```
fields @timestamp, cost_usd, operation
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| stats
    sum(cost_usd) as weekly_cost,
    count(*) as num_calls,
    avg(cost_usd) as avg_cost_per_call
  by bin(@timestamp, 1w), operation
| sort @timestamp desc
| limit 12
```

**Use case:** Track cost trends and growth

---

## 11. Real-Time Cost Monitoring (Last 24 Hours)

Monitor recent costs:

```
fields @timestamp, meetingId, operation, cost_usd, input_tokens, output_tokens, latency_ms
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| filter @timestamp > @timestamp - 24h
| sort @timestamp desc
| limit 100
```

**Use case:** Real-time cost monitoring dashboard

---

## 12. Cost Efficiency Metrics

Calculate cost per processed character/token:

```
fields @timestamp, operation, cost_usd, input_tokens, output_tokens
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| stats
    sum(cost_usd) as total_cost,
    sum(input_tokens + output_tokens) as total_tokens,
    count(*) as num_calls
  by operation
| fields
    operation,
    total_cost,
    total_tokens,
    total_cost / total_tokens * 1000000 as cost_per_1m_tokens,
    total_cost / num_calls as cost_per_call
```

**Use case:** Track cost efficiency over time

---

## 13. Top 10 Most Expensive Meetings

Find the most expensive individual meetings:

```
fields @timestamp, meetingId, cost_usd, operation, input_tokens, output_tokens
| filter message = "LLM_SUMMARY_OK" or message = "CASE_CHECK_LLM_OK"
| stats
    sum(cost_usd) as total_meeting_cost,
    sum(input_tokens) as total_input_tokens,
    sum(output_tokens) as total_output_tokens
  by meetingId
| sort total_meeting_cost desc
| limit 10
```

**Use case:** Identify calls that need optimization (e.g., very long transcripts)

---

## Setting Up CloudWatch Alarms

### Cost Alert: Daily Spend Exceeds Budget

Create a metric filter:
```
Pattern: { $.message = "LLM_SUMMARY_OK" || $.message = "CASE_CHECK_LLM_OK" }
Metric name: BedrockCostUSD
Metric value: $.cost_usd
```

Then create an alarm:
- **Threshold:** Sum > $50 per day (adjust to your budget)
- **Period:** 1 day
- **Action:** Send SNS notification

### Cost Alert: Single Call Too Expensive

Create a metric filter:
```
Pattern: { ($.message = "LLM_SUMMARY_OK" || $.message = "CASE_CHECK_LLM_OK") && $.cost_usd > 0.50 }
Metric name: ExpensiveCall
Metric value: 1
```

Then create an alarm:
- **Threshold:** Sum > 0 (any occurrence)
- **Period:** 5 minutes
- **Action:** Send SNS notification

### Cost Alert: Week-over-Week Increase

Use CloudWatch Anomaly Detection:
- **Metric:** BedrockCostUSD (daily sum)
- **Anomaly detection:** Standard deviation > 2
- **Action:** Send SNS notification when anomaly detected

---

## Example Log Output

Here's what your logs will look like:

```json
{
  "level": "INFO",
  "message": "LLM_SUMMARY_OK",
  "ts": "2025-10-29T14:23:45.123Z",
  "meetingId": "94682083019",
  "operation": "summary",
  "latency_ms": 2847,
  "input_tokens": 12500,
  "output_tokens": 950,
  "total_tokens": 13450,
  "output_chars": 4200,
  "cost_usd": 0.051750,
  "input_cost_usd": 0.037500,
  "output_cost_usd": 0.014250
}
```

```json
{
  "level": "INFO",
  "message": "CASE_CHECK_LLM_OK",
  "ts": "2025-10-29T14:23:48.456Z",
  "meetingId": "94682083019",
  "operation": "case_check",
  "latency_ms": 3214,
  "stop_reason": "end_turn",
  "input_tokens": 17900,
  "output_tokens": 5000,
  "structured_output": true,
  "cost_usd": 0.128700,
  "input_cost_usd": 0.053700,
  "output_cost_usd": 0.075000,
  "cache_read_tokens": 3000,
  "cache_creation_tokens": 0,
  "cache_read_cost_usd": 0.000900,
  "cache_write_cost_usd": 0.000000,
  "cache_savings_usd": 0.008100
}
```

---

## Tips for Cost Optimization

Based on your queries, here's how to optimize:

1. **If you see high input_tokens:**
   - Consider reducing transcript size
   - Enable prompt caching for checklist/KB examples
   - Remove unnecessary context from prompts

2. **If you see high output_tokens:**
   - Simplify output schema (fewer checks)
   - Reduce verbosity in evidence quotes
   - Ask for more concise summaries

3. **If cache_savings_usd is 0:**
   - Enable prompt caching in Bedrock
   - Cache static elements (checklist, KB examples)
   - Can save 8-10% of costs

4. **If latency_ms is high:**
   - Not directly cost-related, but indicates slow processing
   - May want to reduce max_tokens or simplify prompts

5. **If total_cost per meeting > $0.30:**
   - Investigate transcript length (input_tokens)
   - Check if call duration is unusually long
   - Consider transcript preprocessing/summarization

---

## Dashboard Setup

Create a CloudWatch Dashboard with:

1. **Cost widgets:**
   - Line chart: Daily total cost (last 30 days)
   - Pie chart: Cost by operation (summary vs case_check)
   - Number: Month-to-date total cost

2. **Volume widgets:**
   - Line chart: Calls per day
   - Number: Total calls today

3. **Efficiency widgets:**
   - Line chart: Average cost per call over time
   - Line chart: Cache hit rate (if enabled)

4. **Anomaly widgets:**
   - Table: Top 10 most expensive calls today
   - Alarm status: Budget threshold alerts

---

## Questions?

- **How accurate are these costs?** Very accurate - calculated using official AWS Bedrock pricing
- **Does this include AWS Lambda costs?** No, only Bedrock API costs. Lambda costs are typically $0.001-0.002 per call (negligible)
- **What about S3/DynamoDB costs?** Also negligible compared to Bedrock costs (<1%)
- **Can I track costs by customer?** Yes, add customer_id to logs and group by it in queries
