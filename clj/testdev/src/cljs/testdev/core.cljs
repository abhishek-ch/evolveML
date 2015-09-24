(ns testdev.core
    (:require [reagent.core :as reagent :refer [atom]]
              [reagent.session :as session]
              [secretary.core :as secretary :include-macros true]
              [goog.events :as events]
              [goog.history.EventType :as EventType]
              [testdev.contents :as contents]
              )
    (:import goog.History))

;http://www.mattgreer.org/articles/reagent-rocks/
;http://stackoverflow.com/questions/31009978/on-click-handler-for-a-list-item-reagent-clojurescript

(defn loadHome []

  [:header {:class "masthead"}
   [:div {:class "container"}
    [:div {:class "row"}
     [:div {:class "col-sm-6"}
      [:h1
       [:a {:title "Abhishek Testing"} "Abhishek Search..."]
       [:p {:class "lead"} "{An Amazing Company...}"]
       ]]

     [:div {:class "col-sm-6"}
      [:div {:class "pull-right  hidden-xs"}
       [:a {:href "#" :class "dropdown-toggle" :data-toggle "dropdown"}
        [:h3
         [:i {:class "glyphicon glyphicon-cog"}]]]
       [:ul {:class "dropdown-menu"}
        [:li
         [:a {:href "http://www.google.com"}
          [:i {:class "glyphicon glyphicon-chevron-right"}]
          "Link1"]]
        [:li
         [:a {:href "http://www.google.com"}
          [:i {:class "glyphicon glyphicon-user"}]
          "Link2"]]
        [:li
         [:a {:href "http://www.google.com"}
          [:i {:class "glyphicon glyphicon-lock"}]
          "Link3"]]
        [:li
         [:a {:href "http://www.google.com"}
          [:i {:class "glyphicon glyphicon-cog"}]
          "Link4"]]
        ]]]

     ]]]

  )


(defn page [body]
  [:div.page
   [:div {:id "wrap" }
   body
   [:div {:class "divider" :id "section0"}]
   [:div {:class "navbar navbar-custom navbar-inverse navbar-static-top" :id "nav"} (contents/fixed-navigation-bar)]
   [:div {:class "divider" :id "section1"}]
   [:div {:class "container"} (contents/pagecontents)]
    ]
   ]
  )

;;Home Page
(defn home-page []
  (reagent/render-component [page[loadHome]]
                            (.-body js/document))
     )





;; -------------------------
;; Views

(defn home-page1 []
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
