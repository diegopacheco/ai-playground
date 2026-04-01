package com.retirement.config

import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.springframework.context.annotation.{Bean, Configuration}
import org.springframework.http.converter.json.Jackson2ObjectMapperBuilderCustomizer

@Configuration
class JacksonConfig:
  @Bean
  def jacksonScalaModule(): Jackson2ObjectMapperBuilderCustomizer =
    builder => builder.modules(DefaultScalaModule)
