
## Event name : `DCSModuleImpression`
## BDD :
### `DCSModuleImpression`
GIVEN  
I navigate to a any page AND  
one or more widgets are populated with some DCS content after the screen has loaded   
WHEN   
the content of the final widget is known
THEN  
Fire the `DCSModuleImpression` event (listing information about each of the DCS content widgets)


## FIELDS :

  [trexModule.Impressions](../All/TrexDCS.md#Impressions)
  - renderedContentOp.content[i].
    - componentType = `c` 
    - all other DCS impression fields in [TrexDCS](../All/TrexDCS.md#Impressions)
- [List2](../../Connectors/Adobe/List2List1Syntax.md)


## CONNECTORS
### ADOBE :

#### Type : ACTION

Combine the following fields into a single trackAction call:

| Adobe field | Bertie field | Notes |
| --- | --- | --- |
| actionName | `DCSModuleImpression`||
|s.contextData['campaign_content']|`list2`|See [List1 & List2](../../Connectors/Adobe/List2List1Syntax.md)|

Ensure that all other fields contained in [AllActions.md](../All/AllActions.md) are populated


## QA Notes
1. ensure all page information is captured (page_name/catgegory/sub_category/group/type/intent)
2. ensure that all page information is taken from the most recent screen load.
