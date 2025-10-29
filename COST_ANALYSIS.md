# AWS Bedrock Cost Analysis - Call Summariser
## Claude 3.7 Sonnet Pricing

### Pricing (as of 2025)
- **Input tokens:** $3.00 per 1M tokens = **$0.003 per 1K tokens**
- **Output tokens:** $15.00 per 1M tokens = **$0.015 per 1K tokens**
- **Prompt caching (read):** $0.30 per 1M tokens = **$0.0003 per 1K tokens** (90% cheaper)
- **Prompt caching (write):** $3.75 per 1M tokens = **$0.00375 per 1K tokens**

### Regional Availability
- **Your config:** eu-west-2 (London)
- **Note:** Claude 3.7 Sonnet not directly available in eu-west-2 yet
- **Alternative:** Use cross-region inference or nearby regions (eu-west-1 Ireland, eu-central-1 Frankfurt)
- **Pricing:** Consistent across regions for Anthropic models

---

## Cost Per Call Analysis

### Assumptions
- **Call duration:** 45 minutes (typical)
- **Transcript size:** ~50,000 characters = ~12,500 tokens
- **Max call duration:** 1.5 hours = ~90,000 characters = ~22,500 tokens

---

## 1. Summary Generation Costs

### Input Tokens (per call)
- Transcript: 12,500 tokens (45 min call)
- System prompt: ~200 tokens
- User prompt template: ~100 tokens
- **Total input:** ~12,800 tokens

### Output Tokens (per call)
- Summary: ~400 tokens
- Key points (5-7): ~150 tokens
- Action items: ~200 tokens
- Sentiment + themes: ~200 tokens
- **Total output:** ~950 tokens

### Summary Cost Calculation (45 min call)
```
Input:  12,800 tokens × $0.003 = $0.0384
Output:   950 tokens × $0.015 = $0.01425
------------------------------------------
Total per summary:              $0.0527 ≈ $0.05
```

### Summary Cost Range
| Call Duration | Chars | Input Tokens | Output Tokens | Cost |
|--------------|-------|--------------|---------------|------|
| 30 min | 35K | 8,750 | 950 | **$0.040** |
| 45 min | 50K | 12,500 | 950 | **$0.053** |
| 60 min | 70K | 17,500 | 950 | **$0.067** |
| 90 min (max) | 90K | 22,500 | 950 | **$0.082** |

---

## 2. Case Check Costs

### Input Tokens (per call)
- Transcript: 12,500 tokens (45 min call)
- System prompt: ~200 tokens
- Checklist JSON (16 checks): ~2,000 tokens
- KB examples (if enabled): ~3,000 tokens
- User prompt template: ~200 tokens
- **Total input:** ~17,900 tokens

### Output Tokens (per call)
- 16 compliance checks × ~300 tokens = ~4,800 tokens
- Overall stats: ~200 tokens
- **Total output:** ~5,000 tokens

### Case Check Cost Calculation (45 min call)
```
Input:  17,900 tokens × $0.003 = $0.0537
Output:  5,000 tokens × $0.015 = $0.075
------------------------------------------
Total per case check:           $0.1287 ≈ $0.13
```

### Case Check Cost Range
| Call Duration | Chars | Input Tokens | Output Tokens | Cost |
|--------------|-------|--------------|---------------|------|
| 30 min | 35K | 13,750 | 5,000 | **$0.116** |
| 45 min | 50K | 17,900 | 5,000 | **$0.129** |
| 60 min | 70K | 23,500 | 5,000 | **$0.146** |
| 90 min (max) | 90K | 30,500 | 5,000 | **$0.167** |

---

## 3. Complete Processing Cost Per Call

### Combined Cost (Summary + Case Check)
| Call Duration | Summary | Case Check | **Total Cost** | With 20% overhead |
|--------------|---------|------------|----------------|-------------------|
| 30 min | $0.040 | $0.116 | **$0.156** | **$0.187** |
| 45 min (typical) | $0.053 | $0.129 | **$0.182** | **$0.218** |
| 60 min | $0.067 | $0.146 | **$0.213** | **$0.256** |
| 90 min (max) | $0.082 | $0.167 | **$0.249** | **$0.299** |

### **Typical Cost per Call: ~$0.18 - $0.22** 🎯

---

## 4. Monthly Cost Projections

### Scenario A: 100 calls/month (typical)
```
100 calls × $0.218 = $21.80/month
```

### Scenario B: 500 calls/month (moderate)
```
500 calls × $0.218 = $109.00/month
```

### Scenario C: 1,000 calls/month (high volume)
```
1,000 calls × $0.218 = $218.00/month
```

### Scenario D: 5,000 calls/month (enterprise)
```
5,000 calls × $0.218 = $1,090.00/month
```

---

## 5. Cost Breakdown by Component

### Per Typical 45-min Call
```
┌─────────────────┬──────────┬───────────┐
│ Component       │ Tokens   │ Cost      │
├─────────────────┼──────────┼───────────┤
│ Summary Input   │ 12,800   │ $0.038    │
│ Summary Output  │    950   │ $0.014    │
│ Case Input      │ 17,900   │ $0.054    │
│ Case Output     │  5,000   │ $0.075    │
├─────────────────┼──────────┼───────────┤
│ TOTAL           │ 36,650   │ $0.182    │
└─────────────────┴──────────┴───────────┘
```

### Cost Distribution
- **Case Check:** 71% of total cost ($0.129)
- **Summary:** 29% of total cost ($0.053)

**Why Case Check costs more:**
- Larger output (5,000 vs 950 tokens)
- KB examples add input tokens
- More complex checklist structure
- Output tokens are 5× more expensive than input

---

## 6. Cost Optimization Opportunities

### A. Prompt Caching (90% savings on cached input)
**Potential savings if enabled:**
- Checklist JSON (2,000 tokens): Save $0.0054/call
- KB examples (3,000 tokens): Save $0.0081/call
- System prompts (400 tokens): Save $0.0011/call
- **Total savings with caching:** ~$0.015/call (8% reduction)

**With prompt caching:**
- Typical call: $0.182 → **$0.167** 💰

### B. Reduce KB Examples
**Current:** 3,000 tokens of examples
**Optimized:** 1,500 tokens (fewer examples)
- **Savings:** $0.0045/call (2.5% reduction)

### C. Simplify Checklist
**Current:** 16 checks
**If reduced to 12 checks:**
- Output tokens: 5,000 → 3,800
- **Savings:** $0.018/call (10% reduction)

### D. Batch Processing (when available)
**Current status:** Not yet available for Claude 3.7
**Potential savings:** 50% discount on batch API (when released)

---

## 7. Comparison: Before vs After Chunking Removal

### Before (With Chunking - 45 min call)
```
Summary:    1 call   = $0.053
Case Check: 2 chunks = $0.129 × 2 = $0.258
                       ─────────────────────
Total:                          $0.311
```

### After (No Chunking - 45 min call)
```
Summary:    1 call  = $0.053
Case Check: 1 call  = $0.129
                      ──────
Total:                $0.182
```

### **Savings from Removing Chunking: $0.129/call (42% reduction)** 🎉

---

## 8. ROI Analysis

### Manual Processing Cost (Baseline)
- **Time per call:** 30-45 minutes (human review)
- **Hourly rate:** $30/hour (junior analyst)
- **Cost per manual review:** $15-22.50

### Automated Processing (Your System)
- **Cost per call:** $0.18-0.22
- **Time:** 3-5 seconds
- **Savings per call:** **$14.80 - $22.30** 💰

### Break-Even Analysis
```
Development cost: $10,000 (hypothetical)
Savings per call: $20
Break-even point: 500 calls

At 100 calls/month: Break-even in 5 months
At 500 calls/month: Break-even in 1 month
```

---

## 9. Cost Monitoring Recommendations

### Key Metrics to Track
1. **Average tokens per call** (input + output)
2. **Cost per call** (by duration bucket)
3. **Cache hit rate** (if prompt caching enabled)
4. **Monthly spend trend**

### CloudWatch Metrics to Create
```python
# Log these metrics from your Lambda
- "TokensPerCall" (dimension: operation=summary|casecheck)
- "CostPerCall" (dimension: callDuration=30min|45min|60min|90min)
- "CacheHitRate" (if using prompt caching)
- "FailedCalls" (dimension: reason=truncation|timeout|error)
```

### Cost Alert Recommendations
- **Warning:** Monthly spend > $200 (if budget is $250/month)
- **Critical:** Single call > $0.50 (indicates unusually long transcript)
- **Info:** Weekly spend trend increasing >20% week-over-week

---

## 10. Summary

### **Typical Call (45 min) Costs:**
- **Summary:** $0.05
- **Case Check:** $0.13
- **Total:** **$0.18/call**

### **Cost Range:**
- **30 min call:** $0.16/call
- **90 min call:** $0.25/call

### **Monthly Estimates:**
- **100 calls:** $18-22/month
- **500 calls:** $90-110/month
- **1,000 calls:** $180-220/month

### **Cost Optimization Impact:**
- **Removed chunking:** -42% ($0.13/call saved) ✅
- **Prompt caching:** Additional -8% if enabled
- **Total potential savings:** -50% from original architecture

---

## Next Steps

1. **Enable prompt caching** for checklist & KB examples (-8% cost)
2. **Monitor token usage** via CloudWatch logs
3. **Set up cost alerts** in AWS Billing
4. **Consider cross-region inference** if eu-west-2 pricing differs
5. **Track actual vs estimated costs** for the first month

**Questions to consider:**
- What's your monthly call volume target?
- What's your budget allocation for LLM costs?
- Should we implement prompt caching immediately?
