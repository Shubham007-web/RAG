# LangSmith Integration Guide

LangSmith provides observability and evaluation for LLM applications. Track every LLM call, monitor performance, and build evaluation datasets from production queries.

## Quick Setup (5 minutes)

### 1. Create LangSmith Account
- Visit: https://smith.langchain.com/
- Sign up with Google, GitHub, or email
- Create a new organization if prompted

### 2. Get Your API Key
1. Go to your LangSmith Settings: https://smith.langchain.com/settings
2. Click **"API Keys"** in the left sidebar
3. Click **"Create New Key"**
4. Copy your new API key

### 3. Set Up Environment Variables

**Option A: Using .env file (Recommended)**

```bash
# Copy template
cp .env.example .env

# Edit .env
nano .env  # or use your editor
```

Update these lines:
```env
ENABLE_LANGSMITH=true
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGSMITH_PROJECT=rag-chatbot
```

**Option B: Using shell environment**

```bash
export ENABLE_LANGSMITH=true
export LANGSMITH_API_KEY="lsv2_pt_..."
export LANGSMITH_PROJECT=rag-chatbot
```

### 4. Restart the Application

```bash
# Kill existing server if running
# Ctrl+C in the terminal

# Start with tracing enabled
uvicorn rag_api:app --reload --port 8000
```

## Verify It's Working

### In Your Application

When LangSmith is enabled, you'll see this in server logs:
```
LangSmith tracing enabled for project: rag-chatbot
```

### In LangSmith Dashboard

1. Go to https://smith.langchain.com/projects
2. Click on your **"rag-chatbot"** project
3. Make a query to the API:
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the employee discount?"}'
   ```
4. Wait 2-3 seconds, then refresh the LangSmith dashboard
5. You should see a new run appear! 🎉

## What Gets Traced

Each LLM call is automatically logged with:

- **Inputs**: Question + retrieved context
- **Outputs**: Generated answer
- **Latency**: How long the call took
- **Model**: Which LLM was used (qwen:latest)
- **Tokens**: Input/output token counts
- **Status**: Success or error

## Dashboard Features

### 1. Runs Table
- View all LLM calls in chronological order
- Filter by date, duration, status
- Click to see full details

### 2. Run Details
```
Inputs:
  - prompt: "What is the policy?"
  - context: [chunk1, chunk2, ...]

Outputs:
  - answer: "The policy is..."

Latency: 2.34s
Model: qwen:latest
```

### 3. Create Evaluators
Track quality metrics over time:
- Correctness (is answer accurate?)
- Relevance (does answer address question?)
- Grounding (is answer supported by sources?)

### 4. Build Datasets
Export production queries for fine-tuning or evaluation:
```bash
# All queries from last 7 days
# All queries that took >5 seconds
# All queries about "salary"
```

## Advanced: Custom Evaluators

Create evaluators to automatically score answers:

```python
from langsmith import evaluate

def correctness(run, example):
    """Score: 1=correct, 0=incorrect"""
    if "401k" in run.outputs["answer"]:
        return 1
    return 0

def grounding(run, example):
    """Score: how many facts are grounded in sources?"""
    verifications = run.outputs.get("fact_verifications", [])
    verified = sum(1 for v in verifications if v["is_verified"])
    return verified / len(verifications) if verifications else 0

# Run evaluation
evaluate(
    data=dataset,
    evaluators=[correctness, grounding]
)
```

## Troubleshooting

### "LangSmith tracing not working"
1. Check `.env` file has correct API key format
   ```
   LANGSMITH_API_KEY=lsv2_pt_xxxxxxx  # Must start with lsv2_pt_
   ```

2. Verify ENABLE_LANGSMITH=true in .env

3. Restart application

4. Check server logs for errors:
   ```
   grep -i langsmith your.log
   ```

### "Runs not appearing in dashboard"
1. Wait 5-10 seconds after making a query
2. Click "Refresh" in LangSmith UI
3. Check that project name matches: `LANGSMITH_PROJECT=rag-chatbot`
4. Verify API key is valid (try regenerating it)

### "LANGSMITH_API_KEY not found"
1. Make sure `.env` file exists in project root
2. Don't use quotes around the key:
   ```
   # ✓ Correct
   LANGSMITH_API_KEY=lsv2_pt_...
   
   # ✗ Wrong
   LANGSMITH_API_KEY="lsv2_pt_..."
   ```

## Security Notes

- **Never commit `.env`** - it's in .gitignore
- **Don't share your API key** - treat it like a password
- **Rotate keys** if accidentally exposed
- Use different projects for dev/staging/prod:
  ```
  LANGSMITH_PROJECT=rag-chatbot-dev       # for development
  LANGSMITH_PROJECT=rag-chatbot-prod      # for production
  ```

## Pricing

LangSmith has a free tier:
- ✓ First 100 runs per month: free
- ✓ Unlimited dataset creation: free
- ✓ Basic evaluators: free
- Paid plans available for higher volumes

See: https://www.langchain.com/pricing

## Documentation

- Full LangSmith Docs: https://docs.smith.langchain.com/
- LangChain Documentation: https://python.langchain.com/
- Evaluation Patterns: https://docs.smith.langchain.com/evaluation/

## Next Steps

1. ✅ Get API key and set up .env
2. ✅ Restart application
3. ✅ Make a test query
4. ✅ See it appear in LangSmith dashboard
5. 📊 Explore runs and create evaluators
6. 📈 Monitor application performance over time

