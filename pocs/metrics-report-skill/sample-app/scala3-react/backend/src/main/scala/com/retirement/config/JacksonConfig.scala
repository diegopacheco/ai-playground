package com.retirement.config

import com.fasterxml.jackson.databind.Module
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.springframework.context.annotation.{Bean, Configuration}

@Configuration
class JacksonConfig:
  @Bean
  def scalaModule(): Module = DefaultScalaModule
