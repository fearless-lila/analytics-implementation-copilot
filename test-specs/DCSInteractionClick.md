| Titan UK | Titan INT | Mario | Jack's |
| --- | --- | --- | --- |
| yes | yes | no | no |

## Event name : `DCSInteractionClick`

## BDD :
### `DCSInteractionClick.md`
GIVEN  
I am viewing a DCS advert, carousel or stramp/tile  
WHEN   
I click / tap on the DCS item  
THEN  
Fire the `DCSInteractionClick` event


## FIELDS :

  - basicOp
    - component = dcs
    - feature =  <'various'>
    - isDeferred = `true` (note1)
  - [TrexDCS.interactions](../All/TrexDCS.md#Interactions)

## CONNECTORS
### ADOBE :

#### Type : Combine

Combine the following fields into the trackState call of the following screen:

| Adobe field | Bertie field | Note |
| --- | --- | --- |
| s.contextData['content_interaction']= | 1 | |
| s.contextData['page_interaction_element_name']= | <`basicOp.component`>`:`<`basicOp.feature`> | |
| s.contextData['page_interaction_element_type']= | <`basicOp.component`> | |
| s.contextData['campaign_content']= | <`list1 and list2`> | see [List1 & List2](../../Connectors/Adobe/List2List1Syntax.md)|

As well as all other fields in [All Actions](../All/AllActions.md) 

Note:
1. This event is currently tracked as a `non-deferred` event on apps. To reduce server calls, we plan to combine it with the next page load as a `deferred` event. This change is a work in progress.


