from django.test import SimpleTestCase, Client


class HealthTests(SimpleTestCase):
    def setUp(self):
        self.c = Client()

    def test_health_endpoint_returns_200(self):
        r = self.c.get("/health/")
        self.assertEqual(r.status_code, 200)

    def test_health_payload_shape(self):
        r = self.c.get("/health/")
        self.assertEqual(r.json(), {"status": "ok"})

    def test_root_or_app_specific_smoke(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, b"django-python3")
