package com.store.service;

import org.springframework.stereotype.Service;

@Service
public class NotificationService {

    public void confirm(String orderId) {
        System.out.println("order confirmed " + orderId);
    }

    public void sendSms(String phone, String text) {
        System.out.println("sms " + phone + " " + text);
    }
}
