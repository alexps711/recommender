export default interface CEvent {
    event_id: string;
    user_id: string;
    start_time: string;
    city: string;
    state: string;
    country: string;
    lat: number;
    lng: number;
    rating?: number;
}