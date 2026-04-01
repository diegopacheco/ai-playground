package com.retirement

import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication

@SpringBootApplication
class RetirementPlannerApplication

object RetirementPlannerApplication:
  def main(args: Array[String]): Unit =
    SpringApplication.run(classOf[RetirementPlannerApplication], args*)
