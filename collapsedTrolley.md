## Event name: `collapsedTrolley`

## BDD :
### `collapsedTrolley.showAll`
GIVEN that I am on the basket summary screen

AND I have 10 or more products in at least one basket type (groceries, MP or F&F)

WHEN I click on the +# button for the basket

THEN fire `basicOp` event, the first time it's clicked

| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `collapsedTrolley.showAll` | | yes | | |
| basicOp.bertieType | `basicOp` | | yes | | |
| basicOp.component | `trolley` | | yes | v45 | concat component & feature |
| basicOp.feature | `expand icon:`<`ghs`\|`marketplace`\|`ff`>` | | yes | v45 | see above |
| basicOp.isDeferred | false | | yes | | if true, send on next pageload |
|  |  |  |  |  |  |


## `collapsedTrolley.moreInfo`
GIVEN that I am on the basket summary screen

WHEN I click on the allow substitutions more info link

THEN fire `basicOp` event


| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `collapsedTrolley.moreInfo` | | yes | | |
| basicOp.bertieType | `basicOp` | | yes | | |
| basicOp.component | `trolley` | | yes | v45 | concat component & feature |
| basicOp.feature | `more info:`<`ghs`\|`marketplace`\|`ff`>` | | yes | v45 | see above |
| basicOp.isDeferred | false | | yes | | if true, send on next pageload |
|  |  |  |  |  |  |


## `collapsedTrolley.productImageClick`
GIVEN that I am on the basket summary screen

WHEN I click on the allow substitutions more info link

THEN fire `basicOp` event, the first time a product image is clicked


| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `collapsedTrolley.productImageClick` | | yes | | |
| basicOp.bertieType | `basicOp` | | yes | | |
| basicOp.component | `trolley` | | yes | v45 | concat component & feature |
| basicOp.feature | `product image click:`<`ghs`\|`marketplace`\|`ff`>` | | yes | v45 | see above |
| basicOp.isDeferred | false | | false | | if true, send on next pageload |
|  |  |  |  |  |  |
