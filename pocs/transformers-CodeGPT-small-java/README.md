### Result
* Using llamma
* Generating Unit Tests in Java

### Summary Result
```
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}

public void testAddMethod(){
    Calculator calc = new Calculator();
    int result = calc.add(1, 2);
    assertEquals(3, result);
}
```

### Full Result
```
model-00001-of-00002.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 9.98G/9.98G [04:02<00:00, 41.1MB/s]
model-00002-of-00002.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 3.50G/3.50G [01:24<00:00, 41.5MB/s]
Downloading shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [05:27<00:00, 163.94s/it]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:23<00:00, 11.76s/it]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 116/116 [00:00<00:00, 891kB/s]
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

    public class Calculator {
       public int add(int a, int b) {
            return a + b;
        }
    }

    public void testAddMethod(){
        Calculator calc = new Calculator();
        int result = calc.add(1, 2);
        assertEquals(3, result);
    }
```
