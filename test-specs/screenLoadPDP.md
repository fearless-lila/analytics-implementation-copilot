## Event name : `screenLoadPDP`

## BDD :
### `product.detail`
GIVEN That I am seeing the product details screen  

THEN `screenLoadPDP` event fires  

## FIELDS :
### AllPages event
| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | see [AllPages](../All/AllPages.md) | | yes | | |
| pageData.pageTitle | `<product name>` | | yes | pageName, v14, v3 | pageData.pageType:pageData.pageTitle |
| | | | yes | c1, v70 | pageData.pageType |
| | | | yes | c2, v71 | pageData.pageType |
| | | | yes | c3, v72 | pageData.pageType |
| pageData.pageType | `pdp` | | yes | c4, v73 | pageData.pageType |
| pageData.category[i].categoryName | `<Name of the category>` | required by Google Analytics and marketing tags | yes | - | - | 
| pageData.category[i].catFriendlyLevel | `aisle \| department \| superDepartment \| shelf` | required by Google Analytics and marketing tags | yes | - | - | 

**Note:** pageData values here replace the pageData in AllPages - do not create a new pageData event

### renderedContentOp event
| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `product.detail` | renderedContentOp event only | yes | | |
| renderedContentOp.bertieType | `renderedContentOp`| | yes | | | 
| renderedContentOp.content[i].pageType | p | | yes | | | 
| renderedContentOp.content[i].displayArea | grd | | yes | | | 
| renderedContentOp.content[i].panel[i].product.tpnb | `<tpnb>` | | yes | product string | see product string syntax | 
| renderedContentOp.content[i].panel[i].product.tpnc | `<tpnc>` | | yes | | | 
| renderedContentOp.content[i].panel[i].product.gtin | `<gtin>` | | yes | | | 
| renderedContentOp.content[i].panel[i].product.sellerId | <sellerId / 'ghs'> (note 1) | | yes | v106 | see product string syntax | 
| renderedContentOp.content[i].panel[i].product.sellerCategory | <'MPProduct' / 'IGHSProduct' / 'FNFProduct' / 'ProductType'> (note 2) | | yes | v86 | see product string syntax | 
| renderedContentOp.content[i].panel[i].product.isFavourite | boolean | | yes | v128 | see product string syntax | 
| renderedContentOp.content[i].panel[i].product.availability | `<availability>` | | yes | v64 | | 
| renderedContentOp.content[i].panel[i].product.offer[i].offerId | `<offerId>` | do not set offer[i] if product is not on offer | yes | v186, v123 | see product string syntax | 
| renderedContentOp.content[i].panel[i].product.offer[i].proposition | `aldi price match` / `low everyday pricing` | do not set offer[i] if product is not on offer | yes | v169 | see product string syntax | 
| renderedContentOp.content[i].panel[i].product.rating.averageRating | `<average star rating>` | | yes | v20, e38 | x10 so it is an integer |
| renderedContentOp.content[i].panel[i].product.rating.noOfRatings | `<number of ratings>` | | yes | v185, e39 | |
| | | | | v89 | `no` default, see algorithmicProductImpression |
| | | | | v90 | `no` default, see dcsImpression |
| | | | | e3 | | 

## QA Notes
1. When the product has a sellerId on Mango API we pass the value as it is otherwise needs to be equal to ghs or null
2. Mango key product.__typename contains the values "MPProduct", "IGHSProduct", "FNFProduct", and "ProductType" needs to be mapped to product.sellerCategory