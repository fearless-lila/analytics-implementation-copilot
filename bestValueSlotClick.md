## Event name : `bestValueSlotInteraction`

## BDD :
### `bestValueSlotInteraction`

GIVEN I am on the slot page

WHEN I selected the ‘Best value' slot AND click on the "Book Slot" CTA

THEN fire `bestValueSlotInteraction`


## FIELDS :
- basicOp
  - component = `highlight slot`
  - feature = `best value`
  - isDeferred = false
  

## CONNECTORS
### ADOBE :

#### Type : ACTION

Combine these items into the trackState call of the following page

| Adobe field | Bertie field |
| --- | --- |
| s.contextData['content_interaction']= | 1 |
| s.contextData['page_interaction_element_name']= | <`basicOp.component`>`:`<`basicOp.feature`> |
| s.contextData['page_interaction_element_type']= | `<basicOp.component>` | 

[All Actions](../../Connectors/Adobe/AllActions.md)

