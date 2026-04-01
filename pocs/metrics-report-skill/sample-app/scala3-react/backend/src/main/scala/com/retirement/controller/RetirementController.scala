package com.retirement.controller

import com.retirement.model.{RetirementInput, RetirementResult}
import com.retirement.service.RetirementCalculationService
import jakarta.validation.Valid
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.*

@RestController
@RequestMapping(Array("/api/retirement"))
class RetirementController(calculationService: RetirementCalculationService):

  @PostMapping(Array("/calculate"))
  def calculate(@Valid @RequestBody input: RetirementInput): ResponseEntity[?] =
    try
      val result = calculationService.calculate(input)
      ResponseEntity.ok(result)
    catch
      case e: IllegalArgumentException =>
        ResponseEntity.badRequest().body(java.util.Map.of("error", e.getMessage))

  @GetMapping(Array("/health"))
  def health(): ResponseEntity[java.util.Map[String, String]] =
    ResponseEntity.ok(java.util.Map.of("status", "UP"))
