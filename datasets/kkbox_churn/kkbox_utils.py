import ibis


table_schemas = {
    "members": ibis.Schema.from_tuples([
        ('msno', 'string'),
        ('city', 'int32'),
        ('bd', 'int32'),
        ('gender', 'string'),
        ('registered_via', 'int32'),
        ('registration_init_time', 'date'),
    ]),
    "transactions": ibis.Schema.from_tuples([
        ('msno', 'string'),
        ('payment_method_id', 'int32'),
        ('payment_plan_days', 'int32'),
        ('plan_list_price', 'int32'),
        ('actual_amount_paid', 'int32'),
        ('is_auto_renew', 'boolean'),
        ('transaction_date', 'date'),
        ('membership_expire_date', 'date'),
        ('is_cancel', 'boolean'),
    ]),
    "user_logs": ibis.Schema.from_tuples([
        ('msno', 'string'),
        ('date', 'date'),
        ('num_25', 'int32'),
        ('num_50', 'int32'),
        ('num_75', 'int32'),
        ('num_985', 'int32'),
        ('num_100', 'int32'),
        ('num_unq', 'int32'),
        ('total_secs', 'float64'),
    ]),
}