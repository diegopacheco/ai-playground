package com.github.diegopacheco.qdrant.amazon;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ProductCleanerTest {

    private static final String UNIQ_ID = "4C69B61DB1FC16E7013B43FC926E502D";

    private static Map<String, Object> row() {
        Map<String, Object> row = new HashMap<>();
        row.put("Uniq Id", UNIQ_ID);
        row.put("Product Name", "DB Longboards CoreFlex Crossbow 41&quot; Bamboo");
        row.put("Category", "Sports &amp; Outdoors | Outdoor Recreation | Skates");
        row.put("Selling Price", "$237.68");
        row.put("About Product", "Make sure this fits by entering your model number. | RESPONSIVE FLEX | 10 INCH KICKTAILS");
        row.put("Image", "https://images.amazon.com/a.jpg|https://images.amazon.com/b.jpg");
        row.put("Product Url", "https://www.amazon.com/dp/B07KMVJJK7");
        return row;
    }

    @Test
    void plainPriceBecomesANumberSoQdrantCanRangeFilterIt() {
        assertEquals(237.68, ProductCleaner.price("$237.68"));
        assertEquals(1299.0, ProductCleaner.price("$1,299.00"));
    }

    @Test
    void priceRangeUsesTheLowestAdvertisedAmount() {
        assertEquals(10.99, ProductCleaner.price("$10.99 - $15.99"));
    }

    @Test
    void junkOrMissingPriceIsAbsentInsteadOfZeroSoItNeverMatchesACheapFilter() {
        assertNull(ProductCleaner.price("Currently unavailable"));
        assertNull(ProductCleaner.price("Total price: 12"));
        assertNull(ProductCleaner.price(""));
        assertNull(ProductCleaner.price(null));
    }

    @Test
    void topCategoryIsTheFirstUnescapedSegmentSoFacetsGroupWholeDepartments() {
        assertEquals("Sports & Outdoors", ProductCleaner.topCategory("Sports &amp; Outdoors | Outdoor Recreation | Skates"));
        assertEquals("Toys & Games", ProductCleaner.topCategory("Toys &amp; Games"));
    }

    @Test
    void productsWithoutCategoryStayFilterableUnderUncategorized() {
        assertEquals("Uncategorized", ProductCleaner.topCategory(null));
        assertEquals("Uncategorized", ProductCleaner.topCategory("  "));
    }

    @Test
    void htmlEntitiesAreUnescapedSoTextIsEmbeddedAndShownAsWritten() {
        assertEquals("Tom & Jerry 41\" <new>", ProductCleaner.unescape("Tom &amp; Jerry 41&quot; &lt;new&gt;"));
        assertEquals("", ProductCleaner.unescape(null));
    }

    @Test
    void fitBoilerplateIsDroppedBecauseItAddsTheSameNoiseToEveryEmbedding() {
        assertEquals("RESPONSIVE FLEX | 10 INCH KICKTAILS",
                ProductCleaner.about("Make sure this fits by entering your model number. | RESPONSIVE FLEX | 10 INCH KICKTAILS"));
        assertEquals("", ProductCleaner.about("Make sure this fits by entering your model number."));
        assertEquals("", ProductCleaner.about(null));
    }

    @Test
    void aboutKeepsRealBulletsAndUnescapesThem() {
        assertEquals("Fast & light | Ages 3+", ProductCleaner.about("Fast &amp; light |  | Ages 3+"));
    }

    @Test
    void pointIdIsTheDatasetIdAsUuidSoReingestOverwritesInsteadOfDuplicating() {
        assertEquals("4c69b61d-b1fc-16e7-013b-43fc926e502d", ProductCleaner.pointId(UNIQ_ID));
        assertEquals(ProductCleaner.pointId(UNIQ_ID), ProductCleaner.pointId(UNIQ_ID.toLowerCase()));
    }

    @Test
    void nonHexIdsStillMapToAStableValidUuid() {
        String id = ProductCleaner.pointId("sku-123");
        assertEquals(id, ProductCleaner.pointId("sku-123"));
        assertEquals(id, java.util.UUID.fromString(id).toString());
        assertFalse(id.equals(ProductCleaner.pointId("sku-124")));
    }

    @Test
    void firstImageIsTakenFromThePipeSeparatedList() {
        assertEquals("https://images.amazon.com/a.jpg", ProductCleaner.firstImage("https://images.amazon.com/a.jpg|https://images.amazon.com/b.jpg"));
        assertEquals("", ProductCleaner.firstImage(null));
    }

    @Test
    void rowIsCleanedIntoAProduct() {
        Product product = ProductCleaner.fromRow(row()).orElseThrow();
        assertEquals("4c69b61d-b1fc-16e7-013b-43fc926e502d", product.id());
        assertEquals("DB Longboards CoreFlex Crossbow 41\" Bamboo", product.name());
        assertEquals("Sports & Outdoors", product.category());
        assertEquals("Sports & Outdoors | Outdoor Recreation | Skates", product.categoryPath());
        assertEquals(237.68, product.price());
        assertEquals("RESPONSIVE FLEX | 10 INCH KICKTAILS", product.about());
        assertEquals("https://images.amazon.com/a.jpg", product.image());
        assertEquals("https://www.amazon.com/dp/B07KMVJJK7", product.url());
    }

    @Test
    void rowWithoutNameIsSkippedBecauseItCannotBeRecommended() {
        Map<String, Object> missing = row();
        missing.put("Product Name", null);
        Map<String, Object> blank = row();
        blank.put("Product Name", "   ");
        assertTrue(ProductCleaner.fromRow(missing).isEmpty());
        assertTrue(ProductCleaner.fromRow(blank).isEmpty());
    }

    @Test
    void rowWithoutPriceOrCategoryIsStillIndexed() {
        Map<String, Object> row = row();
        row.put("Selling Price", null);
        row.put("Category", null);
        Product product = ProductCleaner.fromRow(row).orElseThrow();
        assertNull(product.price());
        assertEquals("Uncategorized", product.category());
    }

    @Test
    void embeddingTextCarriesNameCategoryAndAboutAndIsCappedForTheEmbeddingModel() {
        Product product = ProductCleaner.fromRow(row()).orElseThrow();
        String text = ProductCleaner.embeddingText(product);
        assertTrue(text.contains(product.name()));
        assertTrue(text.contains("Sports & Outdoors"));
        assertTrue(text.contains("RESPONSIVE FLEX"));
        assertFalse(text.contains("Outdoor Recreation"));

        Product verbose = new Product(product.id(), product.name(), product.category(), product.categoryPath(), null, "x".repeat(5000), "", "");
        assertEquals(ProductCleaner.EMBEDDING_TEXT_LIMIT, ProductCleaner.embeddingText(verbose).length());
    }
}
