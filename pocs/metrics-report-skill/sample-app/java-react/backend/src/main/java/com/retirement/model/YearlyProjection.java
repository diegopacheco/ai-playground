package com.retirement.model;

public class YearlyProjection {

    private int year;
    private int age;
    private double startBalance;
    private double contributions;
    private double interestEarned;
    private double endBalance;

    public YearlyProjection(int year, int age, double startBalance, double contributions, double interestEarned, double endBalance) {
        this.year = year;
        this.age = age;
        this.startBalance = startBalance;
        this.contributions = contributions;
        this.interestEarned = interestEarned;
        this.endBalance = endBalance;
    }

    public int getYear() { return year; }
    public int getAge() { return age; }
    public double getStartBalance() { return startBalance; }
    public double getContributions() { return contributions; }
    public double getInterestEarned() { return interestEarned; }
    public double getEndBalance() { return endBalance; }
}
