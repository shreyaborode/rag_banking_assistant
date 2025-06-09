import unittest
from embeddings.create_embeddings import read_texts_from_folder, create_embedding

class TestEmbeddings(unittest.TestCase):
    def test_read_texts_from_folder(self):
        texts = read_texts_from_folder("data")
        self.assertGreater(len(texts), 0, "No texts found in the folder.")

    def test_create_embedding(self):
        embedding = create_embedding("Test text")
        self.assertEqual(len(embedding), 384, "Embedding size mismatch.")  # Assuming 384 dimensions

if __name__ == "__main__":
    unittest.main()