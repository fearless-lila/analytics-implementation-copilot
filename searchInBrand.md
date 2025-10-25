## Event name : `filterGenericInteraction`

#### JIRA Ticket : `LEGO-46665`

## BDD :
### `searchInBrandImpression`
GIVEN That filter menu is showing to me

WHEN I tap on the 'more brands' or the arrow to land on full brand menu

AND the search in brand feature will appear on the full brand menu

THEN `basicOp` event fires once per session


## FIELDS :
| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `plp.filter generic interaction` | | yes | | |
| basicOp.bertieType | `basicOp` | | yes | | |
| basicOp.component | `filter` | | yes | v45 | concat component & feature |
| basicOp.feature | `bands`:`show more`:`search enabled` | | yes | v45 | add `show more` if user tapped on `show more` button; see above |
| basicOp.isDeferred | false | | yes | | if true, send on next pageload |
| | | | yes | e45 | |



## BDD :
### `searchInBrandSearchBarClick`
GIVEN That search in brand feature is showing to me

WHEN I clicked into the search bar

THEN `basicOp` event fires once per session


## FIELDS :
| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `plp.filter generic interaction` | | yes | | |
| basicOp.bertieType | `basicOp` | | yes | | |
| basicOp.component | `filter` | | yes | v45 | concat component & feature |
| basicOp.feature | `brands`:`search bar` | | yes | v45 | see above |
| basicOp.isDeferred | false | | yes | | if true, send on next pageload |
| | | | yes | e45 | |



## BDD :
### `searchInBrandSearchSuggestion`
GIVEN That search results of brands are showing to me

WHEN I ticked the brand suggestions 

THEN `basicOp` event fires once per session


## FIELDS :
| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `plp.filter generic interaction` | | yes | | |
| basicOp.bertieType | `basicOp` | | yes | | |
| basicOp.component | `filter` | | yes | v45 | concat component & feature |
| basicOp.feature | `brands`:`search suggestion` | | yes | v45 | see above |
| basicOp.isDeferred | true | | yes | | if true, send on next pageload |
| | | | yes | e45 | |




## BDD :
### `searchInBrandError`
GIVEN That search results of brands are showing to me

WHEN no matches error is showing to me

THEN `errorOp` event fires


## FIELDS :
| Bertie Field | Bertie Value | Notes | Required | Adobe Field | Adobe Value / Syntax |
| --- | --- | --- | --- | --- | --- |
| ref | `plp.filter.error` | | yes | | |
| errorOp.bertieType | `errorOp` | | yes | | |
| errorOp.error[i].errorType | `interaction` | | yes | | |
| errorOp.error[i].interactionType | `action` | | yes | | |
| errorOp.error[i].message | <`error message`> | | yes | c38, v48 | |
| errorOp.error[i].code | <`error code if available`> | | yes | | |
| | | | yes | e110 | |