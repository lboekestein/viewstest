## [Feature] Support Distribution Aggregation #41

See original issue on views-platform: https://github.com/views-platform/views-pipeline-core/issues/41

The `AggregationClass` can be found [here](aggregation.py).
Some basic pytests are in the [test_aggregation_manager](test_aggregation_manager.py) file.

For exploring and testing purposes I'm using a [sandbox notebook](sandbox_aggregation_manager.ipynb).

### TODO's
- [X] Implement core AggregationManager class
- [X] Add weighted distribution aggregation
- [ ] Add "concat" aggregation method
- [X] Implement point prediction aggregation
- [ ] Add ensemble statistics calculation
- [ ] Write comprehensive PyTest tests
- [ ] Create documentation and usage examples
- [ ] Performance evaluation on large datasets (Global PGM)