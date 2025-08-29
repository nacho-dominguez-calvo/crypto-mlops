# VPC ID: Para configurar security groups después
output "vpc_id" {
  description = "ID of the VPC where everything lives"
  value       = aws_vpc.main.id
}

# Subnet IDs: Para saber dónde poner ECS tasks
output "public_subnet_ids" {
  description = "Where to put public resources (Load Balancer)"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Where to put private resources (Database, ECS)"
  value       = aws_subnet.private[*].id
}

# Database endpoint: Para conectar desde Python
output "database_endpoint" {
  description = "How to connect to PostgreSQL from your app"
  value       = aws_db_instance.main.endpoint
  sensitive   = true  # No mostrar en logs (contiene info sensible)
}

# S3 bucket names: Para subir/descargar archivos desde Python
output "data_bucket_name" {
  description = "Where to store raw cryptocurrency data"
  value       = aws_s3_bucket.data.bucket
}

output "models_bucket_name" {
  description = "Where to store trained Prophet models"
  value       = aws_s3_bucket.models.bucket
}

# IAM roles: Para que ECS sepa qué permisos tiene
output "ecs_task_role_arn" {
  description = "What permissions ECS containers have"
  value       = aws_iam_role.ecs_task.arn
}
