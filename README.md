# ai-classifier
```markdown
# AI Email Classifier

Go + Python ML service that classifies emails as **Spam**, **Work**, or **Personal**.

## Architecture

```
Client → Go HTTP Server → Python ML Model → Response
                ↓
         Fallback Rules (if ML down)
```

## ML Details

- **Algorithm**: Logistic Regression
- **Features**: TF-IDF vectorizer (unigrams + bigrams)
- **Training data**: 30 labeled emails (expandable)
- **Priority mapping**: Work→high, Personal→medium, Spam→low

## API

```bash
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"urgent meeting at 3pm"}'
```

Response:
```json
{
  "text": "urgent meeting at 3pm",
  "category": "work",
  "priority": "high"
}
```

## Run

```bash
# Terminal 1 - Train & serve ML
python3 ml_model.py && python3 predict.py

# Terminal 2 - Go server
go run main.go
```

## Future Improvements

- Add confidence threshold rejection
- Retrain pipeline with human feedback
- Upgrade to DistilBERT for context understanding
```

---

## You're Ready

You now have:
- ✅ Real ML (not keywords)
- ✅ Working code
- ✅ Something to explain
- ✅ Fallback for reliability

**Walk in tomorrow knowing you built an actual AI system.** The interviewer's job is to verify – not to trick you.

Go get it. 🚀
