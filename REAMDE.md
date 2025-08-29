# 🚀 Crypto MLOps Pipeline

End-to-end MLOps pipeline for cryptocurrency price prediction using AWS and modern DevOps practices.

## 🏗️ Architecture

- **Data Ingestion**: CoinGecko API
- **ML Pipeline**: LSTM model for time series prediction  
- **Infrastructure**: AWS (ECS, RDS, S3, CloudWatch)
- **CI/CD**: GitHub Actions
- **Monitoring**: MLflow + CloudWatch
- **API**: FastAPI for real-time predictions

## 🛠️ Technologies

- **Infrastructure**: Terraform, AWS
- **ML/Data**: Python, TensorFlow, Pandas, MLflow
- **API**: FastAPI, Docker
- **CI/CD**: GitHub Actions
- **Database**: PostgreSQL
- **Monitoring**: CloudWatch, structlog

## 📋 Setup

1. Configure AWS CLI
2. Deploy infrastructure: `cd infrastructure/terraform && terraform apply`
3. Run locally: `docker-compose up`

## 📊 Project Status

- [x] Infrastructure setup
- [ ] Data pipeline
- [ ] ML model training
- [ ] API development
- [ ] CI/CD pipeline
- [ ] Monitoring & alerting

## 👨‍💻 Author

Portfolio project demonstrating MLOps and AWS skills.
EOF