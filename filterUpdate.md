## Event name : `filterUpdate`

## BDD :

### `filter.update`

GIVEN That I am on a plp screen with filter or sort controls

WHEN I change the sort or filter options on the sort/filter overlay either by:
 - changing the sort option
 - adding a filter
 - removing a filter
 - clearing all filters
 
AND I apply the filters

OR I tap X to remove an existing filter lozenge

THEN `filterUpdate` event fires

## FIELDS :

* filterOp
  + interactionType = `manual` || `default` (use 'manual' if filter applied by user, 'default' if filter applied automatically by the site)
  + sortFilter.sortedBy = `default` || `low to high` || `high to low`
  + sortFilter.filter.curated[i].groupName = `brand`
  + sortFilter.filter.curated[i].groupValue = <`brand value`>
  + sortFilter.filter.curated[i].groupName = `category`
  + sortFilter.filter.curated[i].groupValue = <`category name`> 
  + sortFilter.filter.curated[i].groupName = `dietary`  
  + sortFilter.filter.curated[i].groupValue = <`dietary value`> 
  + sortFilter.filter.curated[i].groupName = `Something FF might filter on, e.g. colour`
  + sortFilter.filter.curated[i].groupValue = `The value of the FF filter, e.g. blue`
  + sortFilter.filter.curated[i].groupName = `favourite` (only set if favourite filter is applied)
  + sortFilter.filter.curated[i].groupName = `offer` (only set if offer filter is applied)
  + sortFilter.filter.curated[i].groupName = `new` (only set if new filter is applied)
  + sortFilter.filter.curated[i].groupName = `tesco and marketplace` (not by default)
  + sortFilter.filter.curated[i].groupValue = <`tesco and marketplace | tesco | marketplace`> 
* [AllProductListPages](../All/AllProductListPages.md)  

Note  
1. This is a deferred event  
2. 'applied' refers to all sort and filter options currently in effect on this screenLoad, not those interacted with.
3. When more than one filter of the same type is applied (e.g. 2 brand filters), concatenate the values in groupValue with a pipe e.g. `tesco|finest|pepsi`
4. The names/values capturing in filterOp should not be fixed, should reflects whatever labels showing on the site.

## CONNECTORS

### ADOBE :

#### Type : COMBINE

Combine these items into the resulting trackState call for the relevant PLP

| Adobe field | Bertie field | Note |
| --- | --- |  --- |
| page_filter_action | 1 | event72 |
| number_of_filter_applied | N/A|  (eVar148, prop26) Indicates the number of filters currently applied |
| page_results_filter_display_type_global | `<filterId shorthand code>:<filter status mapping>`  | (eVar147, prop25), see filter status mapping table below, concat value with `\|`, eg:`sor:3\|new:0\|off:0\|bra:0\|pro:1\|cat:1` |
|filter_`<filterId>`_selected  | if sortFilter.filter.curated[i].groupName = `<filterId>` is present then populate with associated sortFilter.filter.curated[i].groupValue | `<filterId>` is not fixed value, should be coming from API. Populate this field only if there is value in `sortFilter.filter.curated[i].groupValue`; see table below for exisiting filter |
| filter_interaction_type | filterOp.interactionType| eVar85 | 

[ProductListPages](../../Connectors/Adobe/ProductListPages.md) 

## Filter Status mapping:
| Filter Status | Adobe | Notes |
| --- | --- | --- |
|false|0|  |
|true|1|  |
|default|2| only apply for sort filter|
|low to high|3| only apply for sort filter||
|high to low|4| only apply for sort filter||

### Adobe Exisiting Filter:
| filterId | shorthand code |s.contextData| Adobe |Notes |
| --- | --- | --- | --- | --- |
|sort|sor|  `['page_results_filter_display_type_global']`| eVar147, prop25
|new|new|  `['page_results_filter_display_type_global']]`| eVar147, prop25
|offer|off| `['page_results_filter_display_type_global']`| eVar147, prop25
|price|pri| `['page_results_filter_display_type_global']`| eVar147, prop25|
|brands|bra| `['filter_brands_selected']` <br>  `['page_results_filter_display_type_global']`| eVar238, eVar147, prop25 |
|productSource|pso|`['filter_marketplace_tesco_selected']` <br>  `['page_results_filter_display_type_global']`| eVar181, eVar147, prop25|
|category|cat| `['filter_categories_selected']` <br>  `['page_results_filter_display_type_global']`| eVar239, eVar147, prop25|
|dietaries|die| `['filter_dietary_selected']` <br>  `['page_results_filter_display_type_global']`| eVar161, eVar147, prop25| 
|favourites|fav|`['page_results_filter_display_type_global']` | eVar147, prop25
|colour|col| `['filter_colour_selected']` <br>  `['page_results_filter_display_type_global']`| eVar16, eVar147, prop25|
|gender|gen| `['filter_gender_selected']` <br>  `['page_results_filter_display_type_global']`| eVar35,eVar147, prop25|
|size|siz| `['filter_size_selected']` <br>  `['page_results_filter_display_type_global']`| eVar39, eVar147, prop25|
|productLength|ple|   `['filter_productLength_selected']` <br>  `['page_results_filter_display_type_global']`|  eVar81, eVar147, prop25|
|sleeveLength|sle|    `['filter_sleeveLength_selected']` <br>  `['page_results_filter_display_type_global']`| eVar83, eVar147, prop25|
|style|sty|    `['filter_style_selected']` <br>  `['page_results_filter_display_type_global']`| eVar84, eVar147, prop25|




## QA NOTES :

1. page_results_filter_display_type_global is pipe delimited and should always have all pieces present in this order. The booleans are powered by the presence of brand, category, dietary and curated filters.
2. The names/values capturing in filterOp should not be fixed, should reflect whatever labels are showing on the site.
3. Adobe Mapping Table needs to be actively updated if any new filters added or filterId value changes from API
4. If the filterId contains two words (e.g., productLength, sleeveLength), we extract the first letter of the first word and the first two letters of the second word. For example, from 'sleeveLength', we get 'sle'. 
