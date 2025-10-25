## Event name : `marketplaceBestsellerFilter`
### Jira Ticket `LEGO-46548`
## BDD :
### `trexImpression` 
GIVEN users are on homepage
OR user interacted with the marketplace best seller filter

WHEN user marketplace bestseller carousel with filter is showing to users

THEN fire renderedContentOp event with below

## FIELDS :

| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `product.algorithmic.impression` | | yes | | |
| renderedContentOp.bertieType | `renderedContentOp` | | yes | | |
| renderedContentOp.content[i].componentType | `t` | |  yes | list1 | 1st place (trex only) |
| renderedContentOp.content[i].componentName | `sugProduct:marketplace bestseller:<filters user clicks on, eg:all,home accessories>` | | yes | Custom Link| |

[algorithmicProductImpression](../../../GHS-web/MFEs/Product%20Tile/algorithmicProductImpression.md)




## BDD :
### `trexCarouselProductClick`
GIVEN marketplace bestseller carousel is showing to users

WHEN user clicks on one of the product in the carousel

THEN fire contentInteractOp bertie event


| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| contentInteractOp.content.componentName | `sugProduct:marketplace bestseller:<filters user clicks on, eg:all,home accessories>` | | yes | Custom Link | |


[algorithmicProductInteraction](../../../GHS-web/MFEs/Product%20Tile/algorithmicProductInteraction.md)


## BDD :
### `trexCarouselBasketInteraction`
GIVEN marketplace bestseller carousel is showing to users

WHEN user clicks on one of the product in the carousel

THEN fire contentInteractOp bertie event


| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| basketOp.identifier | `recommender:trex:marketplace bestseller:<filters user clicks on, eg:all,home accessories>` | | yes | v61 | |


[basketInteraction](../../../GHS-web/MFEs/Basket%20Interactions/basketInteraction.md)



## BDD :
### `carouselTabClick`
GIVEN marketplace bestseller carousel is showing to users

WHEN user interacts with the top tab to load different category

THEN fire basicOp bertie event

## FIELDS :

| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `marketplace bestsellers.change category` | | yes | | |
| basicOp.bertieType | `basicOp` | | yes | | |
| basicOp.component | `marketplace bestsellers` | | yes | v45 | concat component & feature |
| basicOp.feature | `<all / category user clicked on>` | | yes | v45 | see above |
| basicOp.isDeferred | false | | yes | | use `true` if interaction takes to another page |
| | | | yes | e45 | |


## BDD :
### `carouselViewAll`
GIVEN uses are seeing Marketplace bestsellers carousel

WHEN users click on the "view all" CTA

THEN fire basicOp event

## FIELDS :

| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `marketplace bestsellers.change category` | | yes | | |
| basicOp.bertieType | `basicOp` | | yes | | |
| basicOp.component | `marketplace bestsellers` | | yes | v45 | concat component & feature |
| basicOp.feature | `view all` | | yes | v45 | see above |
| basicOp.isDeferred | true | | yes | | use `true` if interaction takes to another page |
| | | | yes | e45 | |


## BDD :
### `carouselArrow`
GIVEN uses are seeing Marketplace bestsellers carousel

WHEN users click on the right/left arrow

THEN fire basicOp event

## FIELDS :

| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `marketplace bestsellers.arrow` | | yes | | |
| basicOp.bertieType | `basicOp` | | yes | | |
| basicOp.component | `marketplace bestsellers` | | yes | v45 | concat component & feature |
| basicOp.feature | `<right/left> arrow` | | yes | v45 | see above |
| basicOp.isDeferred | false | | yes | | use `true` if interaction takes to another page |
| | | | yes | e45 | |