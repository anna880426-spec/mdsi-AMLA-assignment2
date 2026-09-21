# Weather Intelligence API

AI-powered weather prediction service for Sydney, Australia.

## Endpoints
- `GET /` - API overview
- `GET /health` - Health check
- `GET /model-metadata` - Model information
- `POST /predict/index/comfort_climate` - CCI prediction (T+1, T+2, T+3)
- `POST /predict/category/weather_hazard` - WHC prediction (T+7)

## Deployed API
https://mdsi-amla-assignment2.onrender.com/docs

## Installation
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Usage Example
```json
POST /predict/index/comfort_climate
{
  "date": "2025-01-01"
}
```

## Related Repositories
- Experiment Repository: https://github.com/anna880426-spec/mdsi-AMLA-assignment2-Notebook
- Package Repository: https://github.com/anna880426-spec/mdsi-AMLA-assignment2-Package
