package com.retirement.config

import org.springframework.http.ResponseEntity
import org.springframework.web.bind.MethodArgumentNotValidException
import org.springframework.web.bind.annotation.{ExceptionHandler, RestControllerAdvice}

@RestControllerAdvice
class GlobalExceptionHandler:

  @ExceptionHandler(Array(classOf[MethodArgumentNotValidException]))
  def handleValidation(ex: MethodArgumentNotValidException): ResponseEntity[java.util.Map[String, Object]] =
    val errors = new java.util.HashMap[String, String]()
    ex.getBindingResult.getFieldErrors.forEach(err =>
      errors.put(err.getField, err.getDefaultMessage))
    ResponseEntity.badRequest().body(java.util.Map.of("errors", errors))

  @ExceptionHandler(Array(classOf[IllegalArgumentException]))
  def handleIllegalArgument(ex: IllegalArgumentException): ResponseEntity[java.util.Map[String, String]] =
    ResponseEntity.badRequest().body(java.util.Map.of("error", ex.getMessage))
