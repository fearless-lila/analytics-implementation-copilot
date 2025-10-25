| Titan UK | Titan INT | Mario | Jack's |
| --- | --- | --- | --- |
| yes | yes | yes | yes |

## Event name : `trexProductModule`
thie event is used to track the impression of trex products lists
## BDD :
### `trexProductModule`
GIVEN  
I navigate to a any page  
WHEN   
a widget is populated with small number of trex product after the screen has loaded   
THEN  
Fire the `trexProductModule` event


## FIELDS :

  [trexModule](../All/TrexDCS.md#Impressions)
  - renderedContentOp.content[i].
    - componentType = `t` 
    - componentName = <'various'> (trex) **(note 3)** || `offers` (personalised offers)
    - ...
    - componentId = `<strategy id>` 
    - panel[i].posInModule = `<rank>`
    - panel[i].product.tpnb
    - panel[i].product.gtin
    - panel[i].product.sellerId = <sellerId / 'ghs' > (note 5)
    - panel[i].product.personalisationModel = `<model version>-<model name>` (personalised only)
    - panel[i].product.offer[i].offerId 
    - panel[i].product.offer[i].proposition 
    - panel[i].product.sellerCategory = `products[i].__typename` which is available on Mango API response
- [AllProductListActions](../All/AllProductListActions.md)
- [List1 & List 2](../../Connectors/Adobe/List2List1Syntax.md)


## CONNECTORS
### ADOBE :

#### Type : ACTION

Combine the following fields into a single trackAction call:

| Adobe field | Bertie field | Notes |
| --- | --- | --- |
| actionName | `trexProductModule`||
| s.contextData['suggested_product_engine']= |rCO.content[i].componentName |(v166) (note3) |
| s.contextData['suggested_products_module_impression']= | `1` | (e167) (note4) <Only if rCO.ctnt[i].componentType = `t` (trex) &#124;&#124; `c` (dcs)>|
| s.contextData['products'] | | See [Product Syntax](../../Connectors/Adobe/ProductSyntax.md)|
| s.contextData['products'] | product.gtin | (marketplace only) |
| s.contextData['products'] with evar64= | product.availability |  |
| s.contextData['products'] with evar98= | panel[i].posInModule |  |
| s.contextData['products'] with evar106= | product.sellerId | (note 5)  |
| s.contentData['&&products'] with evar86=|`products[i].__typename` which is available on Mango API response|
| s.contextData['products'] with evar88= | panel[i].product.personalisationModel / `no` |  |
| s.contextData['products'] with evar89= | `<pers/no>` | |
| s.contextData['products'] with evar128= | product.isFavourite | <if is true, set `fav`>  |
| s.contextData['products'] with evar186= | product.offer[i].offerId | String of offerIds delimited by ('`~`') (Ex : 8745634`~`8746526) |
| s.contextData['products'] with evar169= |  product.offer[i].proposition |  if no proposition set to 'no' |
|s.contextData['richrelevance_products']|`list2, list1`|See [List1 & List 2](../../Connectors/Adobe/List2List1Syntax.md)|

Ensure
1. product list complies with [AllProductListActions](../All/AllProductListActions.md)
2. all other fields contained in [AllActions.md](../AllActions.md) are populated


## QA Notes
1. ensure all page information is captured (page_name/catgegory/sub_category/group/type/intent)
2. ensure that all page information is taken from the most recent screen load.
3. see below table for current trex `suggested_product_engine` (v166) values.  

| trex carousel | location | rCO/cIO.componentName - v166 |
| --- | --- | --- |
| Special offers - offers chosen for you | homepage | trex-offers |
| pdp carousel | pdp |trex-pdp |
| checkout - interrupt \|\| favourites | various |trex-`<carousel name>` |
| Search - IBYC | search results |trex-sugProduct |
| Browse | aisle or shelves | trex-browse |

4. `suggested_product_module_impression` should only be populated on TREX hits
5. Marketplace products should be identified by Mango key __typename="MPProduct", with sellerId set to the seller uuid. GHS & FF Products should be identified with inverse logic (is not Marketplace), with sellerId set to static value 'ghs'.

