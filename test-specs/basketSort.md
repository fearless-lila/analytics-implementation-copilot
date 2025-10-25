## Event name : `basketSort`

## BDD :
### `basketSortButtonClick`

GIVEN That I am on the basket screen

WHEN I click on the 'sort' button

THEN `basketSortButtonClick` event fires


## FIELDS :
- basicOp
  - component = `basket`
  - feature = `sort`
  - isDeferred = false


## BDD :
### `basketSortOptionSelect`

GIVEN That I clicked on the sort button and sort options is showing

WHEN I click on 'Most recent' or 'Categories'

THEN `basketSortOptionSelect` event fires

## FIELDS :
- basicOp
  - component = `basket`
  - feature = `sort:<most recent | categories>`
  - isDeferred = false



## CONNECTORS
### ADOBE :

#### Type : ACTION


| Adobe field | Bertie field |
| --- | --- |
| s.contextData['content_interaction']= | 1 |
| s.contextData['page_interaction_element_name']= | <`basicOp.component`>`:`<`basicOp.feature`> |
| s.contextData['page_interaction_element_type']= | <`basicOp.component`> |


[All Actions](https://github.dev.global.tesco.org/analytics-capabilities/Web-Implementation/blob/master/tech-specs/GHS-app/specs/Events/All/AllActions.md)

