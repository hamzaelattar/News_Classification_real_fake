from news_classification.data import clean_news_data, load_news_data, split_news_data


def test_local_dataset_pipeline() -> None:
    raw = load_news_data(shuffle=False)
    assert len(raw) == 44_898
    assert set(raw["label"]) == {0, 1}

    cleaned = clean_news_data(raw)
    assert len(cleaned) == 39_103
    assert cleaned.duplicated(subset=["title", "text"]).sum() == 0
    assert cleaned["content"].notna().all()

    train, validation, test = split_news_data(cleaned)
    assert len(train) + len(validation) + len(test) == len(cleaned)
    assert abs(len(train) / len(cleaned) - 0.70) < 0.001
