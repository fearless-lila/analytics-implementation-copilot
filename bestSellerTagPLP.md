## Event name : `bestSellerTag`

## BDD :
### `bestSellerTagImpression`

GIVEN I am on PLP page

WHEN I best seller tag is showing to me

THEN fire `genericImpression` event


## FIELDS :
- basicOp
  - component = `best seller tag`
  - feature = `<number of products on the page with best seller tag>:impression`
  - isDeferred = false
  
[genericImpression](../Generic/genericImpression.md)





## BDD :
### `bestSellerTagBasketInteraction`

GIVEN I am on PLP page and product with best seller tag is showing to me

WHEN I add a product from best seller tag

THEN fire `basketAdd` event


## FIELDS :
- basketOp
  - identifier = `best seller tag`
  
[basketAdd](../Basket%20Interactions/basketAdd.md)