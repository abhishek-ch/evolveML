(ns testdev.core
  (:require [reagent.core :as reagent :refer [atom]]
            [reagent-forms.core :refer [bind-fields init-field value-of]]
            [reagent.session :as session]
            [secretary.core :as secretary :include-macros true]
            [goog.events :as events]
            [goog.history.EventType :as EventType]
            [testdev.contents :as contents]
            [testdev.about :as aboutpage]
            [testdev.example :as example]
            [testdev.register :as register]
            )
  (:import goog.History))

;http://www.mattgreer.org/articles/reagent-rocks/
;http://stackoverflow.com/questions/31009978/on-click-handler-for-a-list-item-reagent-clojurescript
;http://getbootstrap.com/components/

(defn loadHome []

  [:header {:class "masthead"}
   [:div {:class "container"}
    [:div {:class "row"}
     [:div {:class "col-sm-6"}
      [:h1
       [:a {:title "Abhishek Testing"} "Debdoot Agency##..."]
       [:p {:class "lead"} "{A-z of pharmaceutical need}"]
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

;Main page scrollable
(defn page [body]
  [:div.page
   [:div {:id "wrap"}
    body
    [:div {:class "divider" :id "section0"}]
    [:nav {:class "navbar navbar-custom navbar-inverse navbar-static-top" :id "nav" :data-spy "affix" :data-offset-top "100"} (contents/fixed-navigation-bar)]
    [:div {:class "divider" :id "section1"}]
    [:div {:class "container"} (contents/pagecontents)]
    [:div {:class "divider" :id "section2"}]
    [:section {:class "bg-1"} (contents/pageWithImageOne)]
    [:div {:class "divider" :id "section3"}]
    [:section {:class "container" :id "weare"} (contents/pageOwn)]
    [:div {:class "divider"}]
    [:section {:class "bg-5" :id "section4"} (contents/explainDelivery)]
    [:div {:class "container bg-4"} [:div {:class "row"}(contents/delegatePictureBlock)]]
    [:section {:class "divider" :id "section5"}]
    [:div {:class "row"} (contents/direction)]
    [:div {:class "container"} (contents/beforefooter)]
    [:div {:id "footer"} (contents/footer)]
    [:ul {:class "nav pull-right scroll-top"} (contents/pull-up)]
    ]]
  )



(defn friend-source [text]
      (filter
        #(-> % (.toLowerCase %) (.indexOf text) (> -1))
        ["Alice" "Alan" "Bob" "Beth" "Jim" "Jane" "Kim" "Rob" "Zoe"]))

;;refered from https://github.com/jkk/formative
(defn about-page1 []

      [:div
       [:div [:h2 "About testdev"]
        [:div [:a {:href "#/"} "go to the home page"]]
        [:div {:class "container"} (register/forms-example)]
        ]])




;;Home Page
(defn home-page []
      (reagent/render [page [loadHome]]
                      (.-body js/document)))

;;Home Page
(defn home-page-abt []
  (reagent/render [page [loadHome]]
                  (.-body js/document)))


;; -------------------------
;; Views




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
                    (session/put! :current-page #'about-page1))

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
