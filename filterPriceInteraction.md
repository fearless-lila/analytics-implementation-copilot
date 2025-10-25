## Event name : `filterPriceInteraction`

#### JIRA TICKET: `ONA-44269` & `ONA-44275`

## BDD :
### `filterPriceInteractionUpdatePrice`
GIVEN That I entered price manully in the price filter

OR I interacted with the price slider

WHEN I started typing price manully in the price fields OR I started interacting with the price slider

THEN `genericTrackAction` event fires once per page for the same interaction


## FIELDS :
- basicOp
  - component = `filter`
  - feature = `price`:`custom`:`input box/slider`
  - isDeferred = false

Note: Please remove this event if the test is moved to production.

As well as all other fields in [genericTrackAction](../Generic/genericTrackAction.md)
<br>

## BDD :
### `filterExpandInteractionApply`
GIVEN That I interacted with the input box or slider or tickbox in the price filter

WHEN I click on the `show products` button to apply the price filter

THEN `genericInternalLinkDeferredEvent` event fires once per page for the same interaction


## FIELDS :
- basicOp
  - component = `filter`
  - feature = `price`:`custom`:`tickbox`:`apply` - see note 1
  - isDeferred = true
- [filterOp](../Filter/filterUpdate.md)
  - interactionType
  - sortFilter.sortedBy
  - sortFilter.filter.curated[i].groupName 
  - sortFilter.filter.curated[i].groupValue

As well as all other fields in [genericInternalLinkDeferredEvent](../Generic/genericInternalLinkDeferredEvent.md) & [filterUpdate](../Filter/filterUpdate.md)


<br>
Note:

1. use `custom` for input box or slider interaction, use `tickbox` for tickbox interaction, add both if both applied




## CONNECTORS

### ADOBE :
| Adobe field | Bertie field |
| --- | --- |
| s.contextData['content_interaction']= | 1 |
| s.contextData['page_interaction_element_name']= | < `basicOp.component` > `:` < `basicOp.feature` > |
| s.contextData['page_interaction_element_type']= | < `basicOp.component` > |
<br>

## BDD :
### `filterExpandInteractionError`
GIVEN That I am in the processes of interacting filters

WHEN an error message shows up

THEN `errorOp` event fires 

## FIELDS :
  - [AllActions](../All/AllActions.md)
  - errorOp
    - error[i].errorType = `interaction`
    - error[i].interactionType = `action`
    - error[i].message = `<dynamic message>`(please see note 2)
    - error[i].code = `<error code if available>`


#### Type : ACTION

Combine the following fields into a single trackAction call:

| Adobe field | Bertie field |
| --- | --- |
| ActionName | filterPriceError |
| s.contextData['form_errors']= | errorOp.error[i].message |
| s.contextData['error_code']= | errorOp.error[i].code |
| s.contextData['page_error']= | 1 |
| s.contextData['form_submission_error']= | 1 |

As well as all fields in [All Actions](../../Connectors/Adobe/AllActions.md)

### QA Notes
1. Do not be put off by the `form_submission_error` contextData variable (CDV) name.  This CDV is used to populate event11 in Adobe which will become our main error counter.  We need to populate this CDV on all errors and will eventually rename the CDV to something more appropriate at a later stage.
2. If errorData.text.length > 255, errorData.text = errorData.text to 252 length + ’…’
3. This event is different from the web one for tickbox & custom interaction, as in app users are able to apply price filter by both method together.
