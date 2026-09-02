# Terraform for AWS deployment (Phase 19).
# NOT VALIDATED: no AWS credentials/account are available in this sandbox,
# so `terraform plan`/`apply` has not been run. This is real, reviewable
# HCL following standard module patterns, not fabricated to look tested.
# See docs/deployment.md and ADR-008 for the reasoning behind EKS/RDS/S3.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "production"
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "voiceops-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "voiceops-eks"
  cluster_version = "1.29"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    general = {
      instance_types = ["m6i.large"]
      min_size       = 2
      max_size       = 6
      desired_size   = 2
    }
    gpu = {
      instance_types = ["g5.xlarge"] # ASR/GPU inference workloads
      min_size       = 0
      max_size       = 4
      desired_size   = 0
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

resource "aws_db_instance" "postgres" {
  identifier             = "voiceops-postgres"
  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = "db.t3.medium"
  allocated_storage      = 50
  db_name                = "voiceops"
  username               = "voiceops"
  password               = var.db_password
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.this.name
  skip_final_snapshot    = false
}

variable "db_password" {
  type      = string
  sensitive = true
}

resource "aws_db_subnet_group" "this" {
  name       = "voiceops-db-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "rds" {
  name   = "voiceops-rds-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
}

resource "aws_s3_bucket" "audio" {
  bucket = "voiceops-audio-${var.environment}"
}

resource "aws_ecr_repository" "api" {
  name = "voiceops-ai/api"
}

output "eks_cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  value = aws_db_instance.postgres.endpoint
}
