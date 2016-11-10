

// template <typename T>
// ivariable<T>* clip_by_norm (const ivariable<T>* a, T cap) {
// 	if (nullptr == a) return nullptr;
// 	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
// 		[cap](T& collector, T other) {
// 			if (min > other) other = min;
// 			else if (max < other) other = max;
// 			collector = other;
// 		},
// 		[cap](std::vector<ivariable<T>*> args) {
// 			ivariable<T>* a = args.front();
// 			return clip_by_norm(a->get_gradient(), cap);
// 		}, 
// 	nnutils::formatter() << "clip_norm(" << a->get_name() << ")");
// 	return op;
// }