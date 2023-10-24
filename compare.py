from openai.embeddings_utils import cosine_similarity


from embeddings import Embeddings


def compare_embeddings(emb1, emb2):
    """Compare two embeddings."""
    # cos = nn.CosineSimilarity()
    # return cos(torch.tensor([emb1]), torch.tensor([emb2])).item()
    cos = cosine_similarity(emb1, emb2)
    return (cos - 0.7) / 0.3


if __name__ == "__main__":
    emb_service = Embeddings()
    # receives user input
    ref_text = input("Type the reference text: ")
    # creates the embedding for the reference text
    emb1 = emb_service.create(ref_text)
    while True:
        compared_text = input("Type the comparable text: ")
        emb2 = emb_service.create(compared_text)
        print(compare_embeddings(emb1, emb2))
