def assign_attributes(opt, opt_mobilenet, opt_shufflenet):
    for attr, value in opt.__dict__.items():
        setattr(opt_mobilenet, attr, value)
        setattr(opt_shufflenet, attr, value)
