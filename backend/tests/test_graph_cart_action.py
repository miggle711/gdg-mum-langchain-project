import json

from unittest.mock import AsyncMock


def _product_json(product_id="p1"):
    return json.dumps({"id": product_id, "name": "Blue Jacket", "price": 79.99})


def _not_found_json():
    return json.dumps({"error": "Product not found"})


# --- interpret_cart_action (CA) ---

async def test_interpret_cart_action_resolves_reference_to_product_id(mocker):
    import app.graph as graph

    mocker.patch.object(graph, "resolve_product_reference", AsyncMock(return_value="p1"))

    result = await graph.interpret_cart_action({"product_reference": "the blue jacket"})

    graph.resolve_product_reference.assert_called_once_with("the blue jacket")
    assert result == {"resolved_product_id": "p1"}


async def test_interpret_cart_action_without_a_reference_skips_resolution(mocker):
    import app.graph as graph

    resolve_mock = mocker.patch.object(graph, "resolve_product_reference", AsyncMock())

    result = await graph.interpret_cart_action({})

    resolve_mock.assert_not_called()
    assert result == {}


async def test_interpret_cart_action_reference_that_does_not_resolve(mocker):
    import app.graph as graph

    mocker.patch.object(graph, "resolve_product_reference", AsyncMock(return_value=None))

    result = await graph.interpret_cart_action({"product_reference": "a made-up item"})

    assert result == {}


# --- validate_cart_action (CV) ---

async def test_validate_cart_action_valid_add_with_explicit_quantity(mocker):
    import app.graph as graph

    mocker.patch("cart_tools.get_product_impl", AsyncMock(return_value=_product_json()))

    result = await graph.validate_cart_action({
        "resolved_product_id": "p1", "cart_action_type": "add", "quantity": 2,
    })

    assert result == {"cart_action_valid": True, "quantity": 2}


async def test_validate_cart_action_add_without_quantity_defaults_to_one(mocker):
    import app.graph as graph

    mocker.patch("cart_tools.get_product_impl", AsyncMock(return_value=_product_json()))

    result = await graph.validate_cart_action({"resolved_product_id": "p1", "cart_action_type": "add"})

    assert result == {"cart_action_valid": True, "quantity": 1}


async def test_validate_cart_action_remove_ignores_quantity(mocker):
    import app.graph as graph

    mocker.patch("cart_tools.get_product_impl", AsyncMock(return_value=_product_json()))

    result = await graph.validate_cart_action({"resolved_product_id": "p1", "cart_action_type": "remove"})

    assert result == {"cart_action_valid": True}


async def test_validate_cart_action_update_quantity_requires_explicit_quantity(mocker):
    import app.graph as graph

    mocker.patch("cart_tools.get_product_impl", AsyncMock(return_value=_product_json()))

    result = await graph.validate_cart_action({"resolved_product_id": "p1", "cart_action_type": "update_quantity"})

    assert result["cart_action_valid"] is False


async def test_validate_cart_action_update_quantity_with_explicit_quantity_is_valid(mocker):
    import app.graph as graph

    mocker.patch("cart_tools.get_product_impl", AsyncMock(return_value=_product_json()))

    result = await graph.validate_cart_action({
        "resolved_product_id": "p1", "cart_action_type": "update_quantity", "quantity": 5,
    })

    assert result == {"cart_action_valid": True, "quantity": 5}


async def test_validate_cart_action_rejects_zero_or_negative_quantity(mocker):
    import app.graph as graph

    mocker.patch("cart_tools.get_product_impl", AsyncMock(return_value=_product_json()))

    result = await graph.validate_cart_action({
        "resolved_product_id": "p1", "cart_action_type": "add", "quantity": 0,
    })

    assert result["cart_action_valid"] is False


async def test_validate_cart_action_rejects_quantity_over_the_cap(mocker):
    import app.graph as graph

    mocker.patch("cart_tools.get_product_impl", AsyncMock(return_value=_product_json()))

    result = await graph.validate_cart_action({
        "resolved_product_id": "p1", "cart_action_type": "add", "quantity": graph.MAX_CART_ACTION_QUANTITY + 1,
    })

    assert result["cart_action_valid"] is False


async def test_validate_cart_action_rejects_missing_product_id_without_querying_db(mocker):
    import app.graph as graph

    get_product_mock = mocker.patch("cart_tools.get_product_impl", AsyncMock())

    result = await graph.validate_cart_action({"cart_action_type": "add", "quantity": 1})

    get_product_mock.assert_not_called()
    assert result["cart_action_valid"] is False


async def test_validate_cart_action_rejects_product_not_found(mocker):
    import app.graph as graph

    mocker.patch("cart_tools.get_product_impl", AsyncMock(return_value=_not_found_json()))

    result = await graph.validate_cart_action({
        "resolved_product_id": "ghost-id", "cart_action_type": "add", "quantity": 1,
    })

    assert result["cart_action_valid"] is False


async def test_validate_cart_action_rejects_missing_action_type(mocker):
    import app.graph as graph

    mocker.patch("cart_tools.get_product_impl", AsyncMock(return_value=_product_json()))

    result = await graph.validate_cart_action({"resolved_product_id": "p1", "quantity": 1})

    assert result["cart_action_valid"] is False


# --- route_from_cart_validation ---

def test_route_from_cart_validation_routes_to_execute_when_valid():
    import app.graph as graph

    assert graph.route_from_cart_validation({"cart_action_valid": True}) == "execute_cart_action"


def test_route_from_cart_validation_routes_to_clarify_when_invalid():
    import app.graph as graph

    assert graph.route_from_cart_validation({"cart_action_valid": False}) == "clarify_node"
    assert graph.route_from_cart_validation({}) == "clarify_node"


# --- execute_cart_action (EC) ---

async def test_execute_cart_action_add_dispatches_to_add_to_cart_impl(mocker):
    import app.graph as graph

    mock_add = mocker.patch("cart_tools.add_to_cart_impl", AsyncMock(return_value=json.dumps({"message": "Added 2 x Blue Jacket to cart"})))

    state = {"cart_action_type": "add", "resolved_product_id": "p1", "quantity": 2, "session_id": "s1"}
    result = await graph.execute_cart_action(state)

    mock_add.assert_called_once_with("p1", 2, session_id="s1")
    assert result == {"cart_action_result": json.dumps({"message": "Added 2 x Blue Jacket to cart"})}


async def test_execute_cart_action_remove_dispatches_to_remove_from_cart_impl(mocker):
    import app.graph as graph

    mock_remove = mocker.patch("cart_tools.remove_from_cart_impl", AsyncMock(return_value=json.dumps({"message": "Removed item from cart"})))

    state = {"cart_action_type": "remove", "resolved_product_id": "p1", "session_id": "s1"}
    await graph.execute_cart_action(state)

    mock_remove.assert_called_once_with("p1", session_id="s1")


async def test_execute_cart_action_update_quantity_dispatches_to_update_quantity_impl(mocker):
    import app.graph as graph

    mock_update = mocker.patch("cart_tools.update_quantity_impl", AsyncMock(return_value=json.dumps({"message": "Updated quantity to 5"})))

    state = {"cart_action_type": "update_quantity", "resolved_product_id": "p1", "quantity": 5, "session_id": "s1"}
    await graph.execute_cart_action(state)

    mock_update.assert_called_once_with("p1", 5, session_id="s1")


async def test_execute_cart_action_unsupported_action_type_returns_error_without_calling_anything(mocker):
    import app.graph as graph

    mock_add = mocker.patch("cart_tools.add_to_cart_impl", AsyncMock())
    mock_remove = mocker.patch("cart_tools.remove_from_cart_impl", AsyncMock())
    mock_update = mocker.patch("cart_tools.update_quantity_impl", AsyncMock())

    result = await graph.execute_cart_action({"cart_action_type": None, "resolved_product_id": "p1", "session_id": "s1"})

    mock_add.assert_not_called()
    mock_remove.assert_not_called()
    mock_update.assert_not_called()
    assert json.loads(result["cart_action_result"]) == {"error": "Unsupported cart action: None"}


# --- generate_cart_confirmation (CG) ---

def test_generate_cart_confirmation_uses_message_field():
    import app.graph as graph

    state = {"cart_action_result": json.dumps({"message": "Added 2 x Blue Jacket to cart", "items": [], "total": 79.99})}
    assert graph.generate_cart_confirmation(state) == {"response": "Added 2 x Blue Jacket to cart"}


def test_generate_cart_confirmation_falls_back_to_error_field():
    import app.graph as graph

    state = {"cart_action_result": json.dumps({"error": "Item not in cart"})}
    assert graph.generate_cart_confirmation(state) == {"response": "Item not in cart"}


def test_generate_cart_confirmation_falls_back_to_generic_apology():
    import app.graph as graph

    assert graph.generate_cart_confirmation({"cart_action_result": "{}"}) == {
        "response": "Sorry, I couldn't update your cart."
    }
