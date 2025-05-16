import java.util.*;
import java.util.stream.Collectors;

class NaiveBayesClassifier {
    static Map<String, Integer> categoryCounts = new HashMap<>();
    static Map<String, Map<String, Integer>> wordCounts = new HashMap<>();
    static Set<String> vocabulary = new HashSet<>();
    static int totalDocs = 0;

    // Train the classifier
    public static void train(List<Document> trainingData) {
        for (Document doc : trainingData) {
            String category = doc.getCategory();
            categoryCounts.put(category, categoryCounts.getOrDefault(category, 0) + 1);
            totalDocs++;

            wordCounts.putIfAbsent(category, new HashMap<>());
            for (String word : doc.getText().split("\\s+")) {
                vocabulary.add(word);
                Map<String, Integer> wordCategoryMap = wordCounts.get(category);
                wordCategoryMap.put(word, wordCategoryMap.getOrDefault(word, 0) + 1);
            }
        }
    }

    // Predict the category of a given document
    public static String predict(String text) {
        double maxProbability = Double.NEGATIVE_INFINITY;
        String bestCategory = null;

        for (String category : categoryCounts.keySet()) {
            double categoryProbability = Math.log((double) categoryCounts.get(category) / totalDocs);
            double wordProbability = 0.0;

            for (String word : text.split("\\s+")) {
                int wordCount = wordCounts.getOrDefault(category, new HashMap<>()).getOrDefault(word, 0);
                wordProbability += Math.log((wordCount + 1.0) / (vocabulary.size() + totalWordCount(category)));
            }

            double totalProbability = categoryProbability + wordProbability;
            if (totalProbability > maxProbability) {
                maxProbability = totalProbability;
                bestCategory = category;
            }
        }

        return bestCategory;
    }

    // Calculate total word count for a category
    static int totalWordCount(String category) {
        return wordCounts.get(category).values().stream().mapToInt(Integer::intValue).sum();
    }

    // Evaluate the model
    public static void evaluate(List<Document> testData) {
        int correct = 0;
        Map<String, Integer> truePositives = new HashMap<>();
        Map<String, Integer> falsePositives = new HashMap<>();
        Map<String, Integer> falseNegatives = new HashMap<>();

        for (Document doc : testData) {
            String predicted = predict(doc.getText());
            if (predicted.equals(doc.getCategory())) {
                correct++;
                truePositives.put(predicted, truePositives.getOrDefault(predicted, 0) + 1);
            } else {
                falsePositives.put(predicted, falsePositives.getOrDefault(predicted, 0) + 1);
                falseNegatives.put(doc.getCategory(), falseNegatives.getOrDefault(doc.getCategory(), 0) + 1);
            }
        }

        double accuracy = (double) correct / testData.size();
        System.out.println("Accuracy: " + accuracy);

        for (String category : categoryCounts.keySet()) {
            int tp = truePositives.getOrDefault(category, 0);
            int fp = falsePositives.getOrDefault(category, 0);
            int fn = falseNegatives.getOrDefault(category, 0);

            double precision = (double) tp / (tp + fp);
            double recall = (double) tp / (tp + fn);

            System.out.println("Category: " + category);
            System.out.println("Precision: " + precision);
            System.out.println("Recall: " + recall);
        }
    }

    // Main method
    public static void main(String[] args) {
        // Example data
        List<Document> trainingData = Arrays.asList(
                new Document("sports", "football soccer cricket"),
                new Document("technology", "programming java python"),
                new Document("sports", "tennis basketball"),
                new Document("technology", "computers hardware software")
        );

        List<Document> testData = Arrays.asList(
                new Document("sports", "football basketball"),
                new Document("technology", "java programming")
        );

        train(trainingData);
        evaluate(testData);
    }
}

// Helper class to store documents
class Document {
    String category;
    String text;

    public Document(String category, String text) {
        this.category = category;
        this.text = text;
    }

    String getCategory() {
        return category;
    }

    String getText() {
        return text;
    }
}
