#include <jni.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <time.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(XrResult r, const char* op) {
    if (XR_FAILED(r)) throw std::runtime_error(std::string(op) + ": " + std::to_string(r));
}
#define XR_CHECK(call) check((call), #call)
bool tracked(const XrSpaceLocation& p) {
    constexpr XrSpaceLocationFlags required = XR_SPACE_LOCATION_POSITION_VALID_BIT |
        XR_SPACE_LOCATION_ORIENTATION_VALID_BIT | XR_SPACE_LOCATION_POSITION_TRACKED_BIT |
        XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
    const auto& q = p.pose.orientation;
    const auto& v = p.pose.position;
    const float norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    return (p.locationFlags & required) == required && std::isfinite(v.x) && std::isfinite(v.y) &&
        std::isfinite(v.z) && std::isfinite(norm) && norm > 0.99f && norm < 1.01f;
}
void pose(std::ostream& o, const XrPosef& p, float offset = 0) {
    const auto& q = p.orientation;
    // Same local +Z wrist-pivot offset as the browser streamer.
    o << "\"position\":[" << p.position.x + offset * 2 * (q.x*q.z + q.w*q.y)
      << ',' << p.position.y + offset * 2 * (q.y*q.z - q.w*q.x)
      << ',' << p.position.z + offset * (1 - 2*(q.x*q.x + q.y*q.y))
      << "],\"orientation\":[" << q.x << ',' << q.y << ',' << q.z << ',' << q.w << ']';
}
struct Streamer {
    android_app* app;
    XrInstance instance = XR_NULL_HANDLE;
    XrSession session = XR_NULL_HANDLE;
    XrSpace base = XR_NULL_HANDLE, head = XR_NULL_HANDLE;
    XrActionSet actions = XR_NULL_HANDLE;
    XrAction gripPose{}, trigger{}, squeeze{}, stick{}, click{}, primary{}, secondary{};
    std::array<XrPath, 2> hands{};
    std::array<XrSpace, 2> grips{};
    std::array<XrSwapchain, 2> swapchains{};
    std::array<std::vector<XrSwapchainImageOpenGLESKHR>, 2> images;
    std::array<XrViewConfigurationView, 2> viewConfig{};
    GLuint framebuffer = 0;
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
    bool running = false, focused = false, quit = false;
    const char* spaceName = "local";
    int64_t sessionId = 0;
    explicit Streamer(android_app* a) : app(a) {}
    ~Streamer() {
        if (framebuffer) glDeleteFramebuffers(1, &framebuffer);
        for (auto s : swapchains) if (s) xrDestroySwapchain(s);
        for (auto s : grips) if (s) xrDestroySpace(s);
        if (head) xrDestroySpace(head);
        if (base) xrDestroySpace(base);
        if (session) xrDestroySession(session);
        if (actions) xrDestroyActionSet(actions);
        if (instance) xrDestroyInstance(instance);
        if (display != EGL_NO_DISPLAY) {
            eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
            if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
            eglTerminate(display);
        }
    }
    XrPath path(const std::string& s) {
        XrPath p; XR_CHECK(xrStringToPath(instance, s.c_str(), &p)); return p;
    }
    XrAction action(const char* name, XrActionType type) {
        XrActionCreateInfo ci{XR_TYPE_ACTION_CREATE_INFO};
        std::snprintf(ci.actionName, sizeof(ci.actionName), "%s", name);
        std::snprintf(ci.localizedActionName, sizeof(ci.localizedActionName), "%s", name);
        ci.actionType = type; ci.countSubactionPaths = 2; ci.subactionPaths = hands.data();
        XrAction result; XR_CHECK(xrCreateAction(actions, &ci, &result)); return result;
    }
    void init() {
        timespec started{}; clock_gettime(CLOCK_MONOTONIC, &started);
        sessionId = int64_t(started.tv_sec) * 1000000000LL + started.tv_nsec;
        PFN_xrInitializeLoaderKHR initialize = nullptr;
        XR_CHECK(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR", reinterpret_cast<PFN_xrVoidFunction*>(&initialize)));
        XrLoaderInitInfoAndroidKHR loader{XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
        loader.applicationVM = app->activity->vm; loader.applicationContext = app->activity->clazz;
        XR_CHECK(initialize(reinterpret_cast<XrLoaderInitInfoBaseHeaderKHR*>(&loader)));
        XrInstanceCreateInfoAndroidKHR android{XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
        android.applicationVM = app->activity->vm; android.applicationActivity = app->activity->clazz;
        const char* extensions[] = {XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME, XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME};
        XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO}; ci.next = &android;
        std::strcpy(ci.applicationInfo.applicationName, "Openpi Quest Streamer");
        ci.applicationInfo.applicationVersion = 1; ci.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
        ci.enabledExtensionCount = 2; ci.enabledExtensionNames = extensions;
        XR_CHECK(xrCreateInstance(&ci, &instance));
        XrSystemGetInfo si{XR_TYPE_SYSTEM_GET_INFO}; si.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        XrSystemId system; XR_CHECK(xrGetSystem(instance, &si, &system));
        PFN_xrGetOpenGLESGraphicsRequirementsKHR requirements = nullptr;
        XR_CHECK(xrGetInstanceProcAddr(instance, "xrGetOpenGLESGraphicsRequirementsKHR", reinterpret_cast<PFN_xrVoidFunction*>(&requirements)));
        XrGraphicsRequirementsOpenGLESKHR gr{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR};
        XR_CHECK(requirements(instance, system, &gr));
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (!eglInitialize(display, nullptr, nullptr)) throw std::runtime_error("eglInitialize failed");
        const EGLint attrs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE};
        EGLConfig config; EGLint count;
        if (!eglChooseConfig(display, attrs, &config, 1, &count) || !count) throw std::runtime_error("No EGL config");
        const EGLint ctx[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, ctx);
        const EGLint pb[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
        surface = eglCreatePbufferSurface(display, config, pb);
        if (context == EGL_NO_CONTEXT || surface == EGL_NO_SURFACE || !eglMakeCurrent(display, surface, surface, context))
            throw std::runtime_error("EGL context failed");
        XrGraphicsBindingOpenGLESAndroidKHR binding{XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR};
        binding.display = display; binding.config = config; binding.context = context;
        XrSessionCreateInfo sc{XR_TYPE_SESSION_CREATE_INFO}; sc.next = &binding; sc.systemId = system;
        XR_CHECK(xrCreateSession(instance, &sc, &session));
        uint32_t viewCount = 0;
        for (auto& v : viewConfig) v.type = XR_TYPE_VIEW_CONFIGURATION_VIEW;
        XR_CHECK(xrEnumerateViewConfigurationViews(instance, system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
            2, &viewCount, viewConfig.data()));
        if (viewCount != 2) throw std::runtime_error("Expected stereo views");
        uint32_t formatCount = 0;
        XR_CHECK(xrEnumerateSwapchainFormats(session, 0, &formatCount, nullptr));
        std::vector<int64_t> formats(formatCount);
        XR_CHECK(xrEnumerateSwapchainFormats(session, formatCount, &formatCount, formats.data()));
        int64_t colorFormat = 0;
        for (auto f : formats) if (f == GL_SRGB8_ALPHA8 || f == GL_RGBA8) { colorFormat = f; break; }
        if (!colorFormat) throw std::runtime_error("No supported RGBA swapchain format");
        for (int i = 0; i < 2; ++i) {
            XrSwapchainCreateInfo sw{XR_TYPE_SWAPCHAIN_CREATE_INFO};
            sw.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT; sw.format = colorFormat;
            sw.sampleCount = 1; sw.width = viewConfig[i].recommendedImageRectWidth;
            sw.height = viewConfig[i].recommendedImageRectHeight; sw.faceCount = 1; sw.arraySize = 1; sw.mipCount = 1;
            XR_CHECK(xrCreateSwapchain(session, &sw, &swapchains[i]));
            uint32_t n = 0; XR_CHECK(xrEnumerateSwapchainImages(swapchains[i], 0, &n, nullptr));
            images[i].resize(n, {XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR});
            XR_CHECK(xrEnumerateSwapchainImages(swapchains[i], n, &n,
                reinterpret_cast<XrSwapchainImageBaseHeader*>(images[i].data())));
        }
        glGenFramebuffers(1, &framebuffer);
        XrReferenceSpaceCreateInfo rc{XR_TYPE_REFERENCE_SPACE_CREATE_INFO}; rc.poseInReferenceSpace.orientation.w = 1;
        rc.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
        if (XR_FAILED(xrCreateReferenceSpace(session, &rc, &base))) {
            rc.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL; XR_CHECK(xrCreateReferenceSpace(session, &rc, &base));
        } else spaceName = "stage";
        rc.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW; XR_CHECK(xrCreateReferenceSpace(session, &rc, &head));
        hands = {path("/user/hand/left"), path("/user/hand/right")};
        XrActionSetCreateInfo ac{XR_TYPE_ACTION_SET_CREATE_INFO};
        std::strcpy(ac.actionSetName, "teleop"); std::strcpy(ac.localizedActionSetName, "Teleop");
        XR_CHECK(xrCreateActionSet(instance, &ac, &actions));
        gripPose = action("grip_pose", XR_ACTION_TYPE_POSE_INPUT);
        trigger = action("trigger", XR_ACTION_TYPE_FLOAT_INPUT); squeeze = action("squeeze", XR_ACTION_TYPE_FLOAT_INPUT);
        stick = action("stick", XR_ACTION_TYPE_VECTOR2F_INPUT); click = action("stick_click", XR_ACTION_TYPE_BOOLEAN_INPUT);
        primary = action("primary", XR_ACTION_TYPE_BOOLEAN_INPUT); secondary = action("secondary", XR_ACTION_TYPE_BOOLEAN_INPUT);
        std::vector<XrActionSuggestedBinding> bindings;
        for (int i = 0; i < 2; ++i) {
            std::string root = i ? "/user/hand/right/input/" : "/user/hand/left/input/";
            auto add = [&](XrAction a, const char* suffix) { bindings.push_back({a, path(root + suffix)}); };
            add(gripPose, "grip/pose"); add(trigger, "trigger/value"); add(squeeze, "squeeze/value");
            add(stick, "thumbstick"); add(click, "thumbstick/click");
            add(primary, i ? "a/click" : "x/click"); add(secondary, i ? "b/click" : "y/click");
        }
        XrInteractionProfileSuggestedBinding profile{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        profile.interactionProfile = path("/interaction_profiles/oculus/touch_controller");
        profile.countSuggestedBindings = static_cast<uint32_t>(bindings.size()); profile.suggestedBindings = bindings.data();
        XR_CHECK(xrSuggestInteractionProfileBindings(instance, &profile));
        XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO}; attach.countActionSets = 1; attach.actionSets = &actions;
        XR_CHECK(xrAttachSessionActionSets(session, &attach));
        for (int i = 0; i < 2; ++i) {
            XrActionSpaceCreateInfo as{XR_TYPE_ACTION_SPACE_CREATE_INFO}; as.action = gripPose;
            as.subactionPath = hands[i]; as.poseInActionSpace.orientation.w = 1;
            XR_CHECK(xrCreateActionSpace(session, &as, &grips[i]));
        }
        __android_log_print(ANDROID_LOG_INFO, "OpenpiXRStatus", "Initialized; space=%s", spaceName);
    }
    float value(XrAction a, int i) {
        XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO}; info.action = a; info.subactionPath = hands[i];
        XrActionStateFloat state{XR_TYPE_ACTION_STATE_FLOAT}; XR_CHECK(xrGetActionStateFloat(session, &info, &state));
        return state.isActive ? state.currentState : 0;
    }
    bool pressed(XrAction a, int i) {
        XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO}; info.action = a; info.subactionPath = hands[i];
        XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN}; XR_CHECK(xrGetActionStateBoolean(session, &info, &state));
        return state.isActive && state.currentState;
    }
    void emit(XrTime time) {
        XrActiveActionSet active{actions, XR_NULL_PATH};
        XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO}; sync.countActiveActionSets = 1; sync.activeActionSets = &active;
        XrResult synced = xrSyncActions(session, &sync);
        bool usable = focused && synced == XR_SUCCESS;
        XrSpaceLocation viewer{XR_TYPE_SPACE_LOCATION};
        bool viewerTracked = usable && XR_SUCCEEDED(xrLocateSpace(head, base, time, &viewer)) && tracked(viewer);
        std::ostringstream o; o.precision(9);
        timespec now{}; clock_gettime(CLOCK_MONOTONIC, &now);
        const int64_t monotonicNs = int64_t(now.tv_sec) * 1000000000LL + now.tv_nsec;
        timespec wall{}; clock_gettime(CLOCK_REALTIME, &wall);
        const int64_t unixNs = int64_t(wall.tv_sec) * 1000000000LL + wall.tv_nsec;
        const double ms = double(monotonicNs) / 1000000.0;
        o << "{\"type\":\"xr_frame\",\"source\":\"openpi-native\",\"t_client\":" << ms
          << ",\"t_monotonic_ns\":" << monotonicNs << ",\"session\":" << sessionId
          << ",\"t_unix_ns\":" << unixNs
          << ",\"xr_time_ns\":" << time << ",\"focused\":" << (usable ? "true" : "false")
          << ",\"reference_space\":\"" << spaceName << "\",\"viewer\":";
        if (viewerTracked) { o << '{'; pose(o, viewer.pose); o << '}'; } else o << "null";
        o << ",\"controllers\":{"; bool first = true;
        for (int i = 0; i < 2 && viewerTracked; ++i) {
            XrActionStateGetInfo get{XR_TYPE_ACTION_STATE_GET_INFO}; get.action = gripPose; get.subactionPath = hands[i];
            XrActionStatePose state{XR_TYPE_ACTION_STATE_POSE}; XR_CHECK(xrGetActionStatePose(session, &get, &state));
            XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
            if (!state.isActive || XR_FAILED(xrLocateSpace(grips[i], base, time, &location)) || !tracked(location)) continue;
            if (!first) o << ','; first = false;
            o << '"' << (i ? "right" : "left") << "\":{"; pose(o, location.pose, 0.05f);
            o << ",\"tracked\":true,\"valid\":true,\"active\":true,\"buttons\":[";
            float tr = value(trigger, i), sq = value(squeeze, i);
            std::array<float, 6> values{tr, sq, 0, float(pressed(click,i)), float(pressed(primary,i)), float(pressed(secondary,i))};
            for (int j = 0; j < 6; ++j) {
                if (j) o << ',';
                bool p = values[j] > 0.5f;
                o << "{\"p\":" << (p ? "true" : "false") << ",\"t\":" << (p ? "true" : "false") << ",\"v\":" << values[j] << '}';
            }
            get.action = stick; XrActionStateVector2f axes{XR_TYPE_ACTION_STATE_VECTOR2F};
            XR_CHECK(xrGetActionStateVector2f(session, &get, &axes));
            o << "],\"axes\":[0,0," << (axes.isActive ? axes.currentState.x : 0) << ','
              << (axes.isActive ? -axes.currentState.y : 0) << "]}";
        }
        o << "}}";
        __android_log_print(ANDROID_LOG_INFO, "OpenpiXR", "%s", o.str().c_str());
    }
    void events() {
        XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
        while (xrPollEvent(instance, &event) == XR_SUCCESS) {
            if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                auto& e = *reinterpret_cast<XrEventDataSessionStateChanged*>(&event);
                focused = e.state == XR_SESSION_STATE_FOCUSED;
                __android_log_print(ANDROID_LOG_INFO, "OpenpiXRStatus", "Session state %d", int(e.state));
                if (e.state == XR_SESSION_STATE_READY) {
                    XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO}; begin.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    XR_CHECK(xrBeginSession(session, &begin)); running = true;
                } else if (e.state == XR_SESSION_STATE_STOPPING) {
                    running = false; XR_CHECK(xrEndSession(session));
                } else if (e.state == XR_SESSION_STATE_EXITING || e.state == XR_SESSION_STATE_LOSS_PENDING) quit = true;
            } else if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) quit = true;
            event = {XR_TYPE_EVENT_DATA_BUFFER};
        }
    }
    bool render(XrTime time, std::array<XrCompositionLayerProjectionView, 2>& layers) {
        XrViewLocateInfo locate{XR_TYPE_VIEW_LOCATE_INFO}; locate.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        locate.displayTime = time; locate.space = base;
        XrViewState state{XR_TYPE_VIEW_STATE};
        std::array<XrView, 2> views{{{XR_TYPE_VIEW}, {XR_TYPE_VIEW}}}; uint32_t count = 0;
        XR_CHECK(xrLocateViews(session, &locate, &state, 2, &count, views.data()));
        constexpr XrViewStateFlags valid = XR_VIEW_STATE_POSITION_VALID_BIT | XR_VIEW_STATE_ORIENTATION_VALID_BIT;
        if (count != 2 || (state.viewStateFlags & valid) != valid) return false;
        for (int i = 0; i < 2; ++i) {
            XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO}; uint32_t index;
            XR_CHECK(xrAcquireSwapchainImage(swapchains[i], &acquire, &index));
            XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO}; wait.timeout = XR_INFINITE_DURATION;
            XR_CHECK(xrWaitSwapchainImage(swapchains[i], &wait));
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, images[i][index].image, 0);
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) throw std::runtime_error("Incomplete XR framebuffer");
            glViewport(0, 0, viewConfig[i].recommendedImageRectWidth, viewConfig[i].recommendedImageRectHeight);
            glClearColor(0.015f, 0.025f, 0.04f, 1); glClear(GL_COLOR_BUFFER_BIT);
            glBindFramebuffer(GL_FRAMEBUFFER, 0); glFlush();
            XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            XR_CHECK(xrReleaseSwapchainImage(swapchains[i], &release));
            layers[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
            layers[i].pose = views[i].pose; layers[i].fov = views[i].fov;
            layers[i].subImage.swapchain = swapchains[i];
            layers[i].subImage.imageRect.extent = {int32_t(viewConfig[i].recommendedImageRectWidth), int32_t(viewConfig[i].recommendedImageRectHeight)};
        }
        return true;
    }
    void run() {
        init();
        while (!app->destroyRequested && !quit) {
            android_poll_source* source; int events;
            while (ALooper_pollOnce(running ? 0 : 50, nullptr, &events, reinterpret_cast<void**>(&source)) >= 0) {
                if (source) source->process(app, source);
                if (app->destroyRequested) return;
            }
            this->events();
            if (!running) continue;
            XrFrameWaitInfo wait{XR_TYPE_FRAME_WAIT_INFO}; XrFrameState frame{XR_TYPE_FRAME_STATE};
            XR_CHECK(xrWaitFrame(session, &wait, &frame));
            XrFrameBeginInfo begin{XR_TYPE_FRAME_BEGIN_INFO}; XR_CHECK(xrBeginFrame(session, &begin));
            emit(frame.predictedDisplayTime);
            XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO}; end.displayTime = frame.predictedDisplayTime;
            end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
            std::array<XrCompositionLayerProjectionView, 2> views;
            XrCompositionLayerProjection projection{XR_TYPE_COMPOSITION_LAYER_PROJECTION}; projection.space = base;
            projection.viewCount = 2; projection.views = views.data();
            const XrCompositionLayerBaseHeader* layer = reinterpret_cast<XrCompositionLayerBaseHeader*>(&projection);
            if (frame.shouldRender && render(frame.predictedDisplayTime, views)) { end.layerCount = 1; end.layers = &layer; }
            XR_CHECK(xrEndFrame(session, &end));
        }
    }
};
}
void android_main(android_app* app) {
    try { Streamer(app).run(); }
    catch (const std::exception& e) { __android_log_print(ANDROID_LOG_ERROR, "OpenpiXRStatus", "%s", e.what()); }
    ANativeActivity_finish(app->activity);
}
