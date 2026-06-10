# IELTS Study Platform

> Real-time AI-powered IELTS speaking study assistant  
> Built with Azure AI Foundry · Azure Speech · Azure AI Language

<img width="1448" height="901" alt="image" src="https://github.com/user-attachments/assets/7e4f85e6-7df7-49ce-9ac9-70fa34884cf6" />

<img width="1425" height="901" alt="image" src="https://github.com/user-attachments/assets/20b12010-42c9-4eed-b9f4-28d7ff29a284" />

<img width="1424" height="902" alt="image" src="https://github.com/user-attachments/assets/0e23d057-57ae-41e7-be12-334033841a61" />

<img width="731" height="894" alt="image" src="https://github.com/user-attachments/assets/effa9deb-4c1d-4651-8253-1087c3808cba" />

<img width="704" height="897" alt="image" src="https://github.com/user-attachments/assets/c906bd06-5844-4b1b-bfa3-dce004cd8a3e" />

## Features

| 기능 | 기술 |
|---|---|
| 실시간 STT | Azure AI Speech |
| 발음 점수 (정확도/유창성/완성도) | Azure Speech Pronunciation Assessment |
| Target Expression 실시간 감지 + 점수 | Azure Speech + 키워드 매칭 |
| 세션 중 AI 튜터 Q&A | Azure OpenAI gpt-4.1-mini |
| IELTS 4개 채점 기준 피드백 | Azure OpenAI gpt-4.1-mini |
| Band 7-8 모범 답안 생성 | Azure OpenAI gpt-4.1-mini |
| PII 자동 마스킹 (Responsible AI) | Azure AI Language |

## Architecture

```
마이크 입력
  └─ Azure Speech SDK (STT + Pronunciation Assessment)
       └─ FastAPI WebSocket 백엔드
            ├─ Azure AI Language  → PII 마스킹
            ├─ Azure OpenAI       → 피드백 / 문제 생성 / AI 튜터
            └─ React 프론트엔드   → 실시간 대시보드
```

## Tech Stack

`Python` `FastAPI` `WebSocket` `React`  
`Azure AI Foundry` `Azure Speech SDK` `Azure AI Language`

## Run Locally

```bash
# 1. 환경변수 설정
cp .env.example .env
# .env 파일에 Azure 키 입력

# 2. 백엔드
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 3. 프론트엔드 (새 터미널)
cd frontend
python -m http.server 3000
# → http://localhost:3000
```

## .env.example

```
SPEECH_KEY=
SPEECH_REGION=koreacentral
AZURE_OAI_KEY=
AZURE_OAI_ENDPOINT=
AZURE_OAI_MODEL=gpt-4.1-mini
LANGUAGE_KEY=
LANGUAGE_ENDPOINT=
```
