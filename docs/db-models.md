# Database Models

## Category

| Field | Type | Constraints |
| --- | --- | --- |
| `id` | String (UUID) | Primary Key, Auto-generated |
| `name` | String | Not null, Unique |
| `icon` | String | Not null |
| `colorClass` | String | Not null |

**Table:** `categories`

## Product

| Field | Type | Constraints |
| --- | --- | --- |
| `id` | String (UUID) | Primary Key, Auto-generated |
| `name` | String | Not null |
| `price` | Double | Not null |
| `originalPrice` | Double | Optional |
| `rating` | Double | Not null |
| `reviews` | Integer | Not null |
| `image` | String (max 512) | Not null |
| `category_id` | FK → Category | Not null, Eager load |

**Table:** `products`

## Relationships

- Product **many-to-one** Category (via `category_id`)
