# Variable para la región AWS
variable "aws_region" {
  description = "AWS region donde crear los recursos"
  type        = string
  default     = "eu-west-1"
  
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.aws_region))
    error_message = "La región debe tener formato válido (ej: eu-west-1)."
  }
}

# Variable para el ambiente (dev, prod, etc.)
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment debe ser: dev, staging, o prod."
  }
}

# Variable para el nombre del proyecto
variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "crypto-mlops"
  
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name solo puede contener letras minúsculas, números y guiones."
  }
}

# Variable para la password de la base de datos
variable "db_password" {
  description = "Database password for RDS instance"
  type        = string
  sensitive   = true  # No se mostrará en logs
  
  validation {
    condition     = length(var.db_password) >= 8
    error_message = "Password debe tener al menos 8 caracteres."
  }
}
