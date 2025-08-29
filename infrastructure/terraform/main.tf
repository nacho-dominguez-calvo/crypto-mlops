terraform {
  required_version = ">= 1.0"
  
  # Providers necesarios
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
  
  # Backend configuration - donde se guarda el state
  backend "s3" {
    bucket = "mlops-terraform-state-ignacio-480415624749"
    key    = "terraform.tfstate"
    region = "eu-west-1"
  }
}

# Provider de AWS
provider "aws" {
  region = var.aws_region
  
  # Tags por defecto para TODOS los recursos
  default_tags {
    tags = {
      Project     = "crypto-mlops"
      Environment = var.environment
      Owner       = "portfolio"
      ManagedBy   = "terraform"
    }
  }
}
