terraform {
  backend "s3" {
    bucket = "agent-werewolf-tfstate"
    key    = "state/terraform.tfstate"
    region = "us-east-1"
  }
}
