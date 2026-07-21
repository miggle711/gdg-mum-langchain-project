from search import build_product_embedding, es_delete_document, es_upsert_document


async def test_es_upsert_document_indexes_with_doc_id(mock_es):
    doc = {"id": "p1", "name": "Widget", "price": 9.99}

    await es_upsert_document(doc)

    mock_es.index.assert_called_once_with(index="products", id="p1", document=doc)


async def test_es_delete_document_ignores_404(mocker, mock_es):
    # es.options(...) is a sync config-wrapper call, not I/O — only the
    # .delete(...) on its result is actually awaited by es_delete_document.
    mock_scoped_client = mocker.MagicMock()
    mock_scoped_client.delete = mocker.AsyncMock()
    mock_es.options = mocker.MagicMock(return_value=mock_scoped_client)

    await es_delete_document("p1")

    mock_es.options.assert_called_once_with(ignore_status=404)
    mock_scoped_client.delete.assert_called_once_with(index="products", id="p1")


def test_build_product_embedding_uses_document_side_prefix(mocker, mock_embedding_model):
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)

    result = build_product_embedding("Widget", "A great widget")

    encode_call = mock_embedding_model.encode.call_args
    assert encode_call.args[0] == "Represent this product for retrieval: Widget. A great widget"
    assert encode_call.kwargs["normalize_embeddings"] is True
    assert result == [0.1] * 768


def test_build_product_embedding_handles_none_description(mocker, mock_embedding_model):
    mock_embedding_model.encode.return_value = mocker.MagicMock(tolist=lambda: [0.1] * 768)

    build_product_embedding("Widget", None)

    encode_call = mock_embedding_model.encode.call_args
    assert encode_call.args[0] == "Represent this product for retrieval: Widget. "
