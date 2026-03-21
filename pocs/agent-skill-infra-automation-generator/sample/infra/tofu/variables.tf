variable "aws_region" {
  type    = string
  default = "us-east-1"
}
variable "project_name" {
  type    = string
  default = "agent-werewolf"
}
variable "environment" {
  type    = string
  default = "dev"
}
variable "backend_port" {
  type    = number
  default = 3000
}
variable "frontend_port" {
  type    = number
  default = 3001
}
variable "backend_cpu" {
  type    = number
  default = 256
}
variable "backend_memory" {
  type    = number
  default = 512
}
variable "frontend_cpu" {
  type    = number
  default = 256
}
variable "frontend_memory" {
  type    = number
  default = 512
}
