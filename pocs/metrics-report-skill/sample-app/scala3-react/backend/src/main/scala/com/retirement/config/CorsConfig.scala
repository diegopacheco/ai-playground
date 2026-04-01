package com.retirement.config

import org.springframework.context.annotation.{Bean, Configuration}
import org.springframework.web.servlet.config.annotation.{CorsRegistry, WebMvcConfigurer}

@Configuration
class CorsConfig:
  @Bean
  def corsConfigurer(): WebMvcConfigurer =
    new WebMvcConfigurer:
      override def addCorsMappings(registry: CorsRegistry): Unit =
        registry.addMapping("/api/**")
          .allowedOrigins("http://localhost:3000")
          .allowedMethods("GET", "POST", "PUT", "DELETE")
          .allowedHeaders("*")
