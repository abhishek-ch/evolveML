(ns testdev.core
    (:require [reagent.core :as reagent :refer [atom]]
              [reagent.session :as session]
              [secretary.core :as secretary :include-macros true]
              [goog.events :as events]
              [goog.history.EventType :as EventType]
              )
    (:import goog.History))

;; -------------------------
;; Views

(defn home-page []
  [:div [:h2 "Welcome to testdev"]
   [:div [:a {:href "#/about"} "go to about page"]]])

(defn about-page []
  [:div [:h2 "About testdev"]
   [:div [:a {:href "#/"} "go to the home page"]]])


(defn current-page []
  [:div [(session/get :current-page)]])


(defonce selected-department (atom "department!"))

(defn simple-component []
      [:div#sidebar-wrapper
       [:ul.sidebar-nav
        [:li.sidebar-brand [:a {:href "#"} "Departments"]]
        [:li [:a {:on-click #(reset! selected-department "Dairy") :href "#"} "Dairy"]]
        [:li [:a {:on-click #(reset! selected-department "Deli") :href "#"} "Deli"]]
        [:li [:a {:on-click #(reset! selected-department "Grocery") :href "#"} "Grocery"]]]
       [:label @selected-department]])


(defn render-simple []
      (reagent/render-component [simple-component]
                          (.-body js/document)))

;; -------------------------
;; Routes
(secretary/set-config! :prefix "#")

(secretary/defroute "/" []
  (session/put! :current-page #'home-page))

(secretary/defroute "/about" []
  (session/put! :current-page #'about-page))

(secretary/defroute "/render" []
                    (session/put! :current-page #'render-simple))

;; -------------------------
;; History
;; must be called after routes have been defined
(defn hook-browser-navigation! []
  (doto (History.)
    (events/listen
     EventType/NAVIGATE
     (fn [event]
       (secretary/dispatch! (.-token event))))
    (.setEnabled true)))

;; -------------------------
;; Initialize app
(defn mount-root []
  (reagent/render [current-page] (.getElementById js/document "app")))

(defn init! []
  (hook-browser-navigation!)
  (mount-root))
