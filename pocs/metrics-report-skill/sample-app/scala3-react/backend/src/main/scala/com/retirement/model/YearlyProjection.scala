package com.retirement.model

import scala.beans.BeanProperty

class YearlyProjection(
  @BeanProperty var year: Int,
  @BeanProperty var age: Int,
  @BeanProperty var startBalance: Double,
  @BeanProperty var contributions: Double,
  @BeanProperty var interestEarned: Double,
  @BeanProperty var endBalance: Double
)
