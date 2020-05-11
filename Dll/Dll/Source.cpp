#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {
	DLLEXPORT int my_add(int x, int y) {
		return x + y;
	}

	DLLEXPORT int my_mul(int x, int y) {
		return x * y;
	}
}